# Data gathered for the model 📊

The data for the model is gathered purely through the mqtt API key offered by digitransit. The documentation for the API key usage is located [here](https://digitransit.fi/en/developers/apis/5-realtime-api/vehicle-positions/high-frequency-positioning/)

[scraper.py](data_scraping/scraper.py) is the python file running the data gathering. 

### Gathered data points

This is the exact topic used in [scraper.py](data_scraping/scraper.py) for the data gathering: "/hfp/v2/journey/ongoing/+/bus/+/+/+/#" 

These datapoints are being gathered by [scraper.py](data_scraping/scraper.py) from the mqtt API:
- event_type (the type of event transmitted by the bus)
- desi (the route number e.g. 510)
- tst (the date and time)
- lat (latitude coordinate)
- long (longitude coordinate)
- dir (direction of the vehicle, either 1 or 2)
- dl (the actual delay of the vehicle measured by HSL)
- veh (the vehicle id)

All these data points are saved into an SQL database (not in the repo) where the data is then being pulled from for training. The only event types stored are VP (vehicle position), ARS (vehicle has arrived to a stop) and PAS (vehicle passes through a stop without stopping). As seen in [neuralnet.py](neuralnet.py), the dl (delay) is used as the data target y in the training of the model.


### Watcher (now deprecated)

When [scraper.py](data_scraping/scraper.py) was collecting data exclusively from a specific vehicle in a specific route, new data was being gathered **until** that specific vehicle reached the end of its route and stopped transmitting data. Between every hour, [scraper.py](data_scraping/scraper.py) had to be manually restarted with a different vehicle being tracked. [watcher.py](data_scraping/watcher.py) was made to automatize and solve this problem. However, [watcher.py](data_scraping/watcher.py) was no longer needed, when [scraper.py](data_scraping/scraper.py) started to gather data from all routes and all vehicles universally.