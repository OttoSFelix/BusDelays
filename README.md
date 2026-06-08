# BusDelays

This is a project to build a neural network model to predict the delays of different bus routes in the HSL area.

To see which bus routes are included in the model, see [routes.txt](data_scraping/routes.txt)

Here are some of the most important and informative documentation files:
- [Neural network architecture documentation](documentation/nn_arch.md)

To see the accuracy benchmarking with different neural network architectures, see [Neural network benchmarks](documentation/NeuralNet_benchmarks.md).
The accuracy is measured by calculating what percentage of the model predictions in the test data during training have less than 20 seconds of error from the actual delay measured by HSL.

## Tools and techologies used

| ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | ![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) | ![Scikit](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ![Docker](https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white) | ![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-000000?style=flat&logo=python) |