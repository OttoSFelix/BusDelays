# BusDelays

This is a project to build a neural network model to predict the delays of different bus routes in the HSL area.

To see which bus routes are included in the model, see [routes.txt](data_scraping/routes.txt)

Here are some of the most important and informative documentation files:
- [Neural network architecture documentation](documentation/nn_arch.md)
- [Docker services](documentation/docker_services.md)
- [Data gathering](documentation/data_gathering.md)

To see the accuracy benchmarking with different neural network architectures, see [Neural network benchmarks](documentation/NeuralNet_benchmarks.md).
The accuracy is measured by calculating what percentage of the model predictions in the test data during training have less than 20 seconds of error from the actual delay measured by HSL.

## Running the model

After cloning the repository, you need to install Docker (if not already installed) and create the bus_data.db file inside data_scraping/.

After that, to collect data, simply run `docker compose up --build scraper`. 

To train the model on all routes (**Warning, this will overwrite already saved parameters in parameters/**), simply run `docker compose up --build mass-train` (**not yet usable**).

To train the model on a specific route (**This will also overwrite already saved parameters in parameters/ for the specific route**), select which route to train the model on at [neuralnet.py](neuralnet.py#L162), then run `docker compose up --build trainer`.

To run inference on the model, select which route to run inference on at [inference.py](inference.py#L16) and insert the correct topic for that route at [line 13](inference.py#L13), then simply run `docker compose up --build inference`.

## Tools and techologies used

| ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) | ![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) | ![Scikit](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) | ![Docker](https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white) | ![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white) | ![Matplotlib](https://img.shields.io/badge/-Matplotlib-000000?style=flat&logo=python) |