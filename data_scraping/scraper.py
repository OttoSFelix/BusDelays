import paho.mqtt.client as mqtt
import ssl
import json
import sqlite3
import signal
import sys

MQTT_HOST = "mqtt.hsl.fi"
MQTT_PORT = 8883
# /<prefix>/<version>/<journey_type>/<temporal_type>/<event_type>/<transport_mode>/<operator_id>/<vehicle_number>/<route_id>/<direction_id>/<headsign>/<start_time>/<next_stop>/<geohash_level>/<geohash>/<sid>/#
MQTT_TOPIC = '/hfp/v2/journey/ongoing/+/bus/+/+/+/#'
DATATYPES = ['VP', 'ARS', 'PAS']
db_conn = sqlite3.connect('bus_data.db', check_same_thread=False)
db_conn.execute('PRAGMA journal_mode=WAL;')
cursor = db_conn.cursor()

cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS bus_routes (
        route TEXT UNIQUE
    )
''')
cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS vehicle_locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        datatype TEXT,
        time REAL,
        latitude REAL,
        longitude REAL,
        direction TEXT,
        delay REAL,
        route TEXT,
        date TEXT,
        vehicle TEXT,
        lag_delay REAL
    )
''')
db_conn.commit()


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


def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPIC)
    print(f"Subscribed to topic: {MQTT_TOPIC}")


last_known_delays = {"0.0": 0.0, "1.0": 0.0}
last_seen_date = None


def on_message(client, userdata, msg):
    global last_seen_date
    raw_data = msg.payload
    try:
        data = json.loads(raw_data)
        datatype = list(data.keys())[0]
        if datatype in DATATYPES:
            route = data[datatype].get('desi', 0)
            date, time = parse_time(data[datatype]['tst'])
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
            last_known_delays[dir_key] = current_delay

            payload = {
                'datatype': datatype,
                'time': time,
                'latitude': latitude,
                'longitude': longitude,
                'direction': direction,
                'delay': delay,
                'route': route,
                'date': date,
                'vehicle': vehicle,
                'lag_delay': lag_delay
            }

            cursor.execute('INSERT OR IGNORE INTO bus_routes (route) VALUES (?)', (payload['route'], ))

            cursor.execute('''
                INSERT INTO vehicle_locations (datatype, time, latitude, longitude, direction, delay, route, date, vehicle, lag_delay)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (payload['datatype'], payload['time'], payload['latitude'], payload['longitude'], payload['direction'], payload['delay'], payload['route'], payload['date'], payload['vehicle'], payload['lag_delay']))
            db_conn.commit()
    except json.JSONDecodeError:
        print("Failed to decode JSON")

def shutdown_handler(signum, frame):
    print("Received shutdown signal. Disconnecting gracefully...")
    client.disconnect()
    db_conn.close()
    sys.exit(0)


if __name__ == "__main__":
    client = mqtt.Client()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    client.on_connect = on_connect
    client.on_message = on_message

    client.tls_set(tls_version=ssl.PROTOCOL_TLS)

    print(f"Connecting to {MQTT_HOST}:{MQTT_PORT}...")
    client.connect(MQTT_HOST, MQTT_PORT, 60)

    client.loop_forever()
