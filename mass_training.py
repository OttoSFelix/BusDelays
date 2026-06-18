import sqlite3
from neuralnet import NeuralNetwork
import os

def train(route):
    print(f'Beginning training with route {route}')
    network = NeuralNetwork(route)
    network.train()
    network.save_weights()
    print(f'Finished training with route {route}')

def route_iteration():
    db_path = 'data_scraping/bus_data.db' if os.path.exists('data_scraping') else 'bus_data.db'
    db_conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = db_conn.cursor()
    routes = [r[0] for r in cursor.execute('SELECT * FROM bus_routes;').fetchall()]

    for route in routes:
        train(route)

    print('Training is done!!!')

if __name__ == '__main__':
    print('Started training...')
    route_iteration()
