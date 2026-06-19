import torch
import joblib
import paho.mqtt.client as mqtt
import json
import ssl
from neuralnet import PyTorchModel
import math
from time import sleep
import numpy as np

MQTT_HOST = "mqtt.hsl.fi"
MQTT_PORT = 8883
MQTT_TOPIC = '/hfp/v2/journey/ongoing/+/bus/+/+/2510/#'
DATATYPES = ['VP', 'ARS', 'PAS']
RESULTS = []
ROUTE = '510'

print("Loading model and scalers into memory...")
weight_path = f'./parameters/model_route_{ROUTE}.pth'
STATE_DICT = torch.load(weight_path, weights_only=True)

SCALER = joblib.load(f'./parameters/scaler_route_{ROUTE}.joblib')
ENCODER = joblib.load(f'./parameters/encoder_route_{ROUTE}.joblib')
Y_SCALER = joblib.load(f'./parameters/y_scaler_route_{ROUTE}.joblib')

num_unique_buses = STATE_DICT['bus_embedding.weight'].shape[0]

MODEL = PyTorchModel(num_unique_buses=num_unique_buses)
MODEL.load_state_dict(STATE_DICT)
MODEL.eval()
print("Model loaded successfully. Starting stream...")


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

last_known_delays = {"0.0": 0.0, "1.0": 0.0}
last_seen_date = None

def on_message(client, userdata, msg):
    global last_seen_date
    raw_data = msg.payload
    try:
        data = json.loads(raw_data)
        datatype = list(data.keys())[0]
        if datatype in DATATYPES:
            tst_val = data[datatype].get('tst')
            if tst_val is None:
                date, time = "2026-06-16", 0.0
            else:
                date, time = parse_time(tst_val)

            latitude = data[datatype].get('lat', 0)
            longitude = data[datatype].get('long', 0)
            direction = data[datatype].get('dir', 0)
            delay = data[datatype].get('dl', 0)
            vehicle = data[datatype].get('veh', 0)
            if direction == '2':
                direction = 1.0
            else:
                direction = 0.0

            if last_seen_date is not None and date != last_seen_date:
                last_known_delays["0.0"] = 0.0
                last_known_delays["1.0"] = 0.0
            last_seen_date = date

            dir_key = str(direction)
            lag_delay = last_known_delays.get(dir_key, 0.0)

            current_delay = float(delay) if delay is not None else 0.0
            if lag_delay == 0.0:
                last_known_delays[dir_key] = current_delay
            else:
                last_known_delays[dir_key] = (0.9 * lag_delay) + (0.1 * current_delay)

            payload = {'time': time, 'latitude': latitude, 'longitude': longitude, 'direction': direction, 'vehicle': vehicle}
            valid = True
            for item in payload.values():
                if item is None:
                    valid = False
            if valid:
                if delay is not None:
                    RESULTS.append((delay, perform_inference(payload, lag_delay)))

    except json.JSONDecodeError:
        print("Failed to decode JSON")


def perform_inference(data, lag_delay):
    try:
        vehicle_id = ENCODER.transform([int(data['vehicle'])])[0]
    except ValueError:
        vehicle_id = ENCODER.transform([-1])[0]

    time_sec = data['time']
    time_sin = math.sin(time_sec * (2 * math.pi / 86400))
    time_cos = math.cos(time_sec * (2 * math.pi / 86400))

    dummy_input = torch.tensor([[time_sin, time_cos, data['latitude'], data['longitude'], lag_delay, data['direction'], vehicle_id]], dtype=torch.float32)
    dummy_input[:, 2:5] = torch.tensor(SCALER.transform(dummy_input[:, 2:5]), dtype=torch.float32)
    
    with torch.no_grad():
        prediction_scaled = MODEL(dummy_input)

    prediction_unscaled = Y_SCALER.inverse_transform([[prediction_scaled.item()]])

    return prediction_unscaled[0][0]


if __name__ == '__main__':
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.tls_set(tls_version=ssl.PROTOCOL_TLS)

    print(f"Connecting to {MQTT_HOST}:{MQTT_PORT}...")
    client.connect(MQTT_HOST, MQTT_PORT, 60)

    client.loop_start()
    sleep(360)
    client.loop_stop()

    errors = []
    for result in RESULTS:
        if result[1] is not None:
            # print(f'Actual: {result[0]} ----- Predicted: {result[1]:.1f}')
            errors.append(abs(result[0] - result[1]))
    
    print(f'Mean error: {np.mean(errors):.2f}')
    print(f'Sample size: {len(errors)}')
