import torch
import joblib
import paho.mqtt.client as mqtt
import json
import ssl
from neuralnet import PyTorchModel
import math
from time import sleep

MQTT_HOST = "mqtt.hsl.fi"
MQTT_PORT = 8883
MQTT_TOPIC = '/hfp/v2/journey/ongoing/+/bus/+/+/1071/#'
DATATYPES = ['VP', 'ARS', 'PAS']
RESULTS = []

def parse_time(timestamp: str, utc: bool = True, date: bool = True):
    timedata = timestamp
    if date:
        stripped = timedata.split('T')
        stripped_time = stripped[1][:-1]
        stripped_date = stripped[0]
        timedata = stripped_time
        datedata = stripped_date
    time = timedata.split(':')
    h = int(time[0])
    if utc:
        h += 3
    min = int(time[1])
    sec = float(time[2])
    total_time = (h * 60 * 60) + (min * 60) + sec
    return datedata, total_time

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected with result code {reason_code}")
    client.subscribe(MQTT_TOPIC)
    print(f"Subscribed to topic: {MQTT_TOPIC}")

def on_message(client, userdata, msg):
    raw_data = msg.payload
    try:
        data = json.loads(raw_data)
        datatype = list(data.keys())[0]
        if datatype in DATATYPES:
            route = data[datatype].get('desi', 0)
            time = parse_time(data[datatype]['tst'])[1]
            latitude = data[datatype].get('lat', 0)
            longitude = data[datatype].get('long', 0)
            direction = data[datatype].get('dir', 0)
            delay = data[datatype].get('dl', 0)
            vehicle = data[datatype].get('veh', 0)
            if direction == '2':
                direction = 1.0
            else:
                direction = 0.0
            payload = {'time': time, 'latitude': latitude, 'longitude': longitude, 'direction': direction, 'vehicle': vehicle}
            RESULTS.append((delay, load_and_infer(route, payload)))

    except json.JSONDecodeError:
        print("Failed to decode JSON")


def load_and_infer(route, data):
    weight_path = f'./parameters/model_route_{route}.pth'
    state_dict = torch.load(weight_path, weights_only=True)
    num_unique_buses = state_dict['bus_embedding.weight'].shape[0]

    scaler = joblib.load(f'./parameters/scaler_route_{route}.joblib')
    encoder = joblib.load(f'./parameters/encoder_route_{route}.joblib')
    try:
        vehicle_id = encoder.transform([data['vehicle']])[0]
    except ValueError:
        print("Warning: Unseen vehicle ID! Falling back to default vehicle 0.")
        vehicle_id = 0

    time_sec = data['time']
    time_sin = math.sin(time_sec * (2 * math.pi / 86400))
    time_cos = math.cos(time_sec * (2 * math.pi / 86400))

    model = PyTorchModel(num_unique_buses=num_unique_buses)
    model.load_state_dict(state_dict)
    model.eval()
    dummy_input = torch.tensor([[time_sin, time_cos, data['latitude'], data['longitude'], data['direction'], vehicle_id]], dtype=torch.float32)
    dummy_input[:, 2:4] = torch.tensor(scaler.transform(dummy_input[:, 2:4]), dtype=torch.float32)
    with torch.no_grad():
        prediction = model(dummy_input)

    return prediction.item()


if __name__ == '__main__':
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    client.on_connect = on_connect
    client.on_message = on_message

    client.tls_set(tls_version=ssl.PROTOCOL_TLS)


    print(f"Connecting to {MQTT_HOST}:{MQTT_PORT}...")
    client.connect(MQTT_HOST, MQTT_PORT, 60)

    client.loop_start()

    sleep(20)

    client.loop_stop()

    for result in RESULTS:
        print(f'{result[0]} ----- {result[1]}')


# {"VP":{"desi":"554","dir":"2","oper":12,"veh":2603,"tst":"2026-06-03T18:50:48.955Z","tsi":1780512648,"spd":11.25,"hdg":231,"lat":60.211535,"long":25.080516,"acc":0.11,"dl":-168,"odo":25814,"drst":0,"oday":"2026-06-03","jrn":721,"line":263,"start":"21:02","loc":"GPS","stop":1453118,"route":"2554","occu":0}}