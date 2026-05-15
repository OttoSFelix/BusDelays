import paho.mqtt.client as mqtt
import ssl
import json
import sqlite3

MQTT_HOST = "mqtt.hsl.fi"
MQTT_PORT = 8883
# /<prefix>/<version>/<journey_type>/<temporal_type>/<event_type>/<transport_mode>/<operator_id>/<vehicle_number>/<route_id>/<direction_id>/<headsign>/<start_time>/<next_stop>/<geohash_level>/<geohash>/<sid>/#
MQTT_TOPIC = "/hfp/v2/journey/+/+/+/+/01438/1056/#"
DATATYPES = ['VP', 'ARS', 'PAS']
VEHICLES = set()
db_conn = sqlite3.connect('bus_data.db', check_same_thread=False)
cursor = db_conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS vehicle_locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datatype TEXT,
        time REAL,
        latitude REAL,
        longitude REAL,
        direction TEXT,
        delay REAL
    )
''')
db_conn.commit()


def parse_time(timestamp: str, utc: bool = True, date: bool = True):
    timedata = timestamp
    if date:
        stripped = timedata.split('T')[1]
        stripped = stripped[:-1]
        timedata = stripped
    time = timedata.split(':')
    h = int(time[0])
    if utc:
        h += 3
    min = int(time[1])
    sec = float(time[2])
    total = (h * 60 * 60) + (min * 60) + sec
    return total


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)
    print(f"Subscribed to topic: {MQTT_TOPIC}")


def on_message(client, userdata, msg):
    raw_data = msg.payload
    try:
        data = json.loads(raw_data)
        datatype = list(data.keys())[0]
        if datatype in DATATYPES:
            time = parse_time(data[datatype]['tst'])
            latitude = data[datatype].get('lat', 0)
            longitude = data[datatype].get('long', 0)
            direction = data[datatype].get('dir', 0)
            delay = data[datatype].get('dl', 0)
            if direction == 2:
                direction = 1.0
            else:
                direction = 0.0
            payload = {'datatype': datatype, 'time': time, 'latitude': latitude, 'longitude': longitude, 'direction': direction, 'delay': delay}
            cursor.execute('''
                INSERT INTO vehicle_locations (datatype, time, latitude, longitude, direction, delay)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (payload['datatype'], payload['time'], payload['latitude'], payload['longitude'], payload['direction'], payload['delay']))
            db_conn.commit()
    except json.JSONDecodeError:
        print("Failed to decode JSON")


if __name__ == "__main__":
    client = mqtt.Client()

    client.on_connect = on_connect
    client.on_message = on_message

    client.tls_set(tls_version=ssl.PROTOCOL_TLS)

    try:
        print(f"Connecting to {MQTT_HOST}:{MQTT_PORT}...")
        client.connect(MQTT_HOST, MQTT_PORT, 60)

        client.loop_forever()
    except KeyboardInterrupt:
        print("Disconnecting...")
        client.disconnect()
        db_conn.close()
