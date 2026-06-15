# Docker services 🐳

Here is the explanation and the use case for every service in [compose.yaml](../compose.yaml)

### scraper

This service collects data from the mqtt API key provided by digitransit to train the neural network model. This service uses [scraper.py](../data_scraping/scraper.py) as its main script and uses a volume to store data into bus_data.db (ignored by git). The dockerfile for scraper is [Dockerfile](../Dockerfile)
More about data gathering for the project and [scraper.py](../data_scraping/scraper.py) can be read from [data_gathering](data_gathering.md).

run scraper with:
`docker compose up --build scraper`

### trainer

This service is used to train the model. It fetches data from bus_data.db and completes the full training with one docker command. This service uses [neuralnet.py](../neuralnet.py) as its main script and uses a volume to fetch data from bus_data.db (ignored by git). The dockerfile for scraper is [Dockerfile.training](../Dockerfile.training) More about the training and [neuralnet.py](../neuralnet.py) can be read from [Neural network architecture documentation](nn_arch.dm).

run trainer with:
`docker compose up --build trainer`

### watcher

This is a deprecated service. It was used to automatize switching between which vehicle scraper was tracking and gathering data from at the moment. More about watcher can be read from [data gathering](data_gathering.md#Watcher-(now-deprecated))

### inference

This service is used to test the inference of the model against real-time data from a scpecific route. This service uses [inference.py](../inference.py) as its main script and uses the parameters folder as a volume to load the model weights, encoder and scaler based on the route to perform the inference. It outputs the mean error from the actual delay measured by HSL. This service will be improved.

run inference with:
`docker compose up --build inference`
