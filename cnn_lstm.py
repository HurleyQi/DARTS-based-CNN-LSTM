from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from darts import TimeSeries

from darts.logging import get_logger, raise_if_not
from darts.models.forecasting.pl_forecasting_module import (
    PLPastCovariatesModule,
    io_processor,
)
from darts.models.forecasting.torch_forecasting_model import PastCovariatesTorchModel
from darts.utils.torch import MonteCarloDropout

logger = get_logger(__name__)

ACTIVATIONS = [
    "ReLU",
    "RReLU",
    "PReLU",
    "ELU",
    "Softplus",
    "Tanh",
    "SELU",
    "LeakyReLU",
    "Sigmoid",
    "GELU",
]

class _CNNLSTM(PLPastCovariatesModule):
    def __init__(
        self,
        input_dim: int,  # Input sequence length for past covariates
        output_dim: int,  # Forecasting horizon
        nr_params: int,

        num_cnn_layers: int,   # Number of CNN layers
        pooling_kernel_size: Tuple[int],  # Pooling kernel sizes
        cnn_kernel_size: Tuple[int],  # Kernel sizes for CNN layers
        cnn_out_channels: Tuple[int],  # Output channels for CNN
        
        num_lstm_layers: int,  # LSTM layers
        lstm_hidden_size: int,  # Hidden state size of LSTM
        
        batch_norm: bool,
        dropout_prob: float,  # Dropout probability
        activation: str,  # Activation function
        max_pool: bool,     # Whether to use max pooling
        **kwargs,  # Allow for other keyword arguments
    ):
        super().__init__(**kwargs)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.nr_params = nr_params
        self.input_chunk_length_multi = self.input_chunk_length * input_dim
        self.output_chunk_length_multi = self.output_chunk_length * output_dim

        self.activation = getattr(nn, activation)()

        self.num_cnn_layers = num_cnn_layers
        self.pooling_kernel_size = pooling_kernel_size
        self.max_pool = max_pool
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_out_channels = cnn_out_channels
        self.dropout_prob = dropout_prob
        self.num_lstm_layers = num_lstm_layers
        self.lstm_hidden_size = lstm_hidden_size

        # CNN layers
        layers = []
        layers.append(
            nn.Conv1d(in_channels=input_dim, out_channels=self.cnn_out_channels[0], 
                      kernel_size=self.cnn_kernel_size[0], padding=1)
        )
        layers.append(self.activation)
        for i in range(1, self.num_cnn_layers):
            layers.append(nn.Conv1d(in_channels=self.cnn_out_channels[i-1], 
                                    out_channels=self.cnn_out_channels[i], 
                                    kernel_size=self.cnn_kernel_size[i], padding=1))
            layers.append(self.activation)
            if self.max_pool:
                layers.append(nn.MaxPool1d(kernel_size=self.pooling_kernel_size[i]))
            else:
                layers.append(nn.AvgPool1d(kernel_size=self.pooling_kernel_size[i]))
            if self.dropout_prob > 0:
                layers.append(nn.Dropout(p=self.dropout_prob))
        self.cnn_layers = nn.Sequential(*layers)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.cnn_out_channels[-1],
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.num_lstm_layers)

        # Output layer
        self.ln = nn.Linear(in_features=self.lstm_hidden_size, out_features=self.output_chunk_length_multi)

    @io_processor
    def forward(self, x_in: Tuple):
        x, _ = x_in
        x = torch.reshape(x, (x.shape[0], self.input_chunk_length_multi, 1))
        # squeeze last dimension (because model is univariate)
        # x = x.squeeze(dim=2)
        
        batch_size, seq_len, input_dim = x.size()
        x = x.view(batch_size, input_dim, seq_len)  # Reshape for CNN
        x = self.cnn_layers(x)
        x = x.permute(0, 2, 1)  # Reshape for LSTM (batch_size, seq_len, num_features)
        lstm_out, _ = self.lstm(x)

        predictions = self.ln(lstm_out[:, -1, :])  # Use the last output of the LSTM
        predictions = predictions.view(
            predictions.shape[0], self.output_chunk_length, self.input_dim, self.nr_params
        )[:, :, : self.output_dim, :]
        
        return predictions


    # @io_processor       
    # def forward(self, x_in):
    #     x, _ = x_in
        
    #     # Reshape the input to fit CNN (batch_size, input_chunk_length_multi, 1)
    #     x = torch.reshape(x, (x.shape[0], self.input_chunk_length_multi, 1))
        
    #     # Prepare for CNN (batch_size, input_dim, seq_len)
    #     batch_size, seq_len, input_dim = x.size()
    #     x = x.view(batch_size, input_dim, seq_len)
        
    #     # Apply CNN layers
    #     x = self.cnn_layers(x)
        
    #     # Prepare for LSTM (batch_size, seq_len, num_features)
    #     x = x.permute(0, 2, 1)
    #     lstm_out, _ = self.lstm(x)
        
    #     # Linear layer to get the final predictions
    #     predictions = self.ln(lstm_out[:, -1, :])  # Use the last LSTM output

    #     predictions = predictions.view(
    #         predictions.shape[0], 
    #         self.output_chunk_length,  # Corresponding to output length in Darts
    #         self.input_dim,  # Number of features or channels
    #         self.nr_params  # If using more parameters (like multivariate)
    #     )[:, :, : self.output_dim, :]  # Keep only target dimensions if needed

    #     return predictions


class CNNLSTM(PastCovariatesTorchModel):
    def __init__(
        self, 
        input_chunk_length: int,
        output_chunk_length: int,
        num_cnn_layers: int,
        cnn_kernel_size: Union[int, Tuple[int]], 
        cnn_out_channels: Union[int, Tuple[int]], 
        pooling_kernel_size: Union[int, Tuple[int]], 
        dropout_prob: float, 
        num_lstm_layers: int,
        lstm_hidden_size: int,
        activation: str = "ReLU",
        max_pool: bool = True, 
        output_chunk_shift: int = 0,
        **kwargs
    ):
        raise_if_not(
            activation in ACTIVATIONS, f"'{activation}' is not in {ACTIVATIONS}"
        )

        raise_if_not(
            isinstance(pooling_kernel_size, int) or len(pooling_kernel_size) == num_cnn_layers,
            "`pooling_kernel_size` is not an int or not a list of ints with length equaling the number of CNN layers",
            logger,
        )
        raise_if_not(
            isinstance(cnn_kernel_size, int) or len(cnn_kernel_size) == num_cnn_layers,
            "`cnn_kernel_size` is not an int or not a list of ints with length equaling the number of CNN layers",
            logger,
        )
        raise_if_not(
            isinstance(cnn_out_channels, int) or len(cnn_out_channels) == num_cnn_layers,
            "`cnn_out_channels` is not an int or not a list of ints with length equaling the number of CNN layers",
            logger,
        )

        super().__init__(**self._extract_torch_model_params(**self.model_params))
        self.pl_module_params = self._extract_pl_module_params(**self.model_params)


        # Handle single integer inputs by expanding to list
        if isinstance(pooling_kernel_size, int):
            pooling_kernel_size = [pooling_kernel_size] * num_cnn_layers
        if isinstance(cnn_kernel_size, int):
            cnn_kernel_size = [cnn_kernel_size] * num_cnn_layers
        if isinstance(cnn_out_channels, int):
            cnn_out_channels = [cnn_out_channels] * num_cnn_layers

        self.num_cnn_layers = num_cnn_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_out_channels = cnn_out_channels
        self.pooling_kernel_size = pooling_kernel_size
        self.lstm_hidden_size = lstm_hidden_size
        self.dropout_prob = dropout_prob
        self.max_pool = max_pool
        self.num_lstm_layers = num_lstm_layers
        self.activation = activation

    @property
    def supports_multivariate(self) -> bool:
        return True

    def _create_model(self, train_sample: Tuple[torch.Tensor]) -> torch.nn.Module:
        # samples are made of (past_target, past_covariates, future_target)
        input_dim = train_sample[0].shape[1] + (train_sample[1].shape[1] if train_sample[1] is not None else 0)
        output_dim = train_sample[-1].shape[1]
        # nr_params = 1 if self.likelihood is None else self.likelihood.num_parameters
        nr_params = 1 

        return _CNNLSTM(
            input_dim=input_dim,
            output_dim=output_dim,
            nr_params=nr_params,
            num_cnn_layers=self.num_cnn_layers,
            cnn_kernel_size=self.cnn_kernel_size,
            cnn_out_channels=self.cnn_out_channels,
            pooling_kernel_size=self.pooling_kernel_size,
            lstm_hidden_size=self.lstm_hidden_size,
            dropout_prob=self.dropout_prob,
            batch_norm=False,
            max_pool=self.max_pool,
            activation=self.activation,
            num_lstm_layers=self.num_lstm_layers,
            **self.pl_module_params
        )




