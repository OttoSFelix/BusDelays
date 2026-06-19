import sqlite3
import paho.mqtt.client as mqtt
import ssl
import json
from time import sleep
import os
import subprocess

MQTT_HOST = "mqtt.hsl.fi"
MQTT_PORT = 8883
MQTT_TOPIC = "/hfp/v2/journey/+/+/+/+/+/1056/#"
MESSAGE_DATA = []

db_path = 'data_scraping/bus_data.db' if os.path.exists('data_scraping') else 'bus_data.db'
db_conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = db_conn.cursor()

def on_connect(client, userdata, flags, rc):
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    raw_data = msg.payload
    data = json.loads(raw_data)
    datatype = list(data.keys())[0]
    start = parse_time(data[datatype]['start'])
    vehicle = data[datatype]['veh']

    MESSAGE_DATA.append((start, vehicle))

def parse_time(time):
    splitted = time.split(':')
    hours = int(splitted[0])
    minutes = int(splitted[1])
    total = (hours * 60 * 60) + (minutes * 60)
    return total

def change_vehicle():
    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.tls_set(tls_version=ssl.PROTOCOL_TLS)
    client.connect(MQTT_HOST, MQTT_PORT, 60)
    client.loop_start()
    sleep(5)
    client.loop_stop()

    MESSAGE_DATA.sort(key=lambda x: x[0], reverse=True)

    new_vehicle = vehicle_number(MESSAGE_DATA[0][1])
    new_topic = f"MQTT_TOPIC = '/hfp/v2/journey/+/+/+/+/{new_vehicle}/1056/#'"
    print(f'Starter tracking new vehicle: {new_vehicle}')
    edit_line(new_topic)

def vehicle_number(vehicle):
    new_vehicle = '0' * (5 - len(str(vehicle)))
    new_vehicle += str(vehicle)
    return new_vehicle

def edit_line(new_topic):
    scraper_path = os.path.join(os.path.dirname(__file__), 'scraper.py')
    with open(scraper_path, 'r') as file:
        lines = file.readlines()

    with open(scraper_path, 'w') as file:
        for line in lines:
            if line.startswith("MQTT_TOPIC ="):
                file.write(f"{new_topic}\n")
            else:
                file.write(line)


if __name__ == '__main__':
    sleep(60)
    tag1 = cursor.execute('SELECT COUNT(*) FROM vehicle_locations;').fetchone()[0]
    sleep(5)
    tag2 = cursor.execute('SELECT COUNT(*) FROM vehicle_locations;').fetchone()[0]

    if int(tag1) == int(tag2):

        change_vehicle()

        subprocess.run(["docker", "restart", "scraper"], check=True)