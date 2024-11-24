# Convolution Long Short-Term Memory Networks (CNN-LSTM)

This repository contains code for a Convolution Long Short-Term Memory Networks (CNN-LSTM) 
deep learning model. 

## Overview 

Although CNN are traditionally used in computer vision for image related tasks, they can also be applied to a wide range of other challenges [1]. CNN utilizes convolution to captures local features and pooling to prevent redundancy. A stack of LSTMs is then connected to the end of CNN with the hopes of capturing sequential relationships. 

We utilized the [DARTS](https://unit8co.github.io/darts/) [2] to  enhance functionality and streamline development. Our model can be
used in similar form as all other DARTS Forecasting Models. 

## Contact
If you have any questions, please email hurleyqi@utexas.edu

## References
[1] Li, Z., Liu, F., Yang, W., Peng, S., & Zhou, J. (2021). A survey of convolutional neural networks: analysis, applications, and prospects. IEEE transactions on neural networks and learning systems, 33(12), 6999-7019.

[2] Herzen, J., Lässig, F., Piazzetta, S. G., Neuer, T., Tafti, L., Raille, G., ... & Grosch, G. (2022). Darts: User-friendly modern machine learning for time series. Journal of Machine Learning Research, 23(124), 1-6.