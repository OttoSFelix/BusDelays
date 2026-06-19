import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from skorch.dataset import ValidSplit
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import sqlite3
import math
import os
import joblib
import random

class PyTorchModel(nn.Module):
    def __init__(self, num_unique_buses):
        super().__init__()
        self.bus_embedding = nn.Embedding(num_embeddings=num_unique_buses, embedding_dim=16)

        self.main_network = nn.Sequential(
            nn.Linear(22, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, X):
        continuous_features = X[:, :6]
        bus_ids = X[:, 6].long()
        embedded_buses = self.bus_embedding(bus_ids)
        combined_features = torch.cat((continuous_features, embedded_buses), dim=1)
        return self.main_network(combined_features)

class NeuralNetwork:
    def __init__(self, route):
        db_path = 'data_scraping/bus_data.db' if os.path.exists('data_scraping') else 'bus_data.db'
        self.db_conn_data = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor_data = self.db_conn_data.cursor()
        self.route = route

        self.weights = []

    def train(self, graph: bool = False):
        print('Preparing data...')
        data = self.cursor_data.execute('''
            SELECT id, datatype, time, latitude, longitude, direction, delay, route, date, vehicle, lag_delay FROM vehicle_locations
            WHERE route = ?
            AND delay IS NOT NULL
            AND time IS NOT NULL
            AND latitude IS NOT NULL
            AND longitude IS NOT NULL
            AND direction IS NOT NULL
            AND vehicle IS NOT NULL
            AND lag_delay IS NOT NULL
            ORDER BY id ASC
        ''', (self.route,)).fetchall()

        raw_vehicle_ids = [int(n[9]) for n in data]

        raw_vehicle_ids.append(-1)

        self.encoder = LabelEncoder()
        encoded_vehicle_ids = self.encoder.fit_transform(raw_vehicle_ids)
        num_unique_buses = len(self.encoder.classes_)

        unknown_token_index = self.encoder.transform([-1])[0]

        encoded_vehicle_ids = encoded_vehicle_ids[:-1]

        X_data = []
        for i, n in enumerate(data):
            time_sec = n[2]
            time_sin = math.sin(time_sec * (2 * math.pi / 86400))
            time_cos = math.cos(time_sec * (2 * math.pi / 86400))
            latitude = n[3]
            longitude = n[4]
            lag_delay = float(n[10])
            direction = float(n[5])
            vehicle = encoded_vehicle_ids[i]
            if random.random() < 0.05:
                vehicle = unknown_token_index
            X_data.append([time_sin, time_cos, latitude, longitude, lag_delay, direction, vehicle])

        X = torch.tensor(X_data, dtype=torch.float32)
        y = torch.tensor([n[6] for n in data], dtype=torch.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        print(f'train sizes: {X_train.size(), y_train.size()}')
        print(f'test sizes: {X_test.size(), y_test.size()}')

        self.scaler = StandardScaler()
        X_train_scaled = X_train.clone()
        X_test_scaled = X_test.clone()
        X_train_scaled[:, 2:5] = torch.tensor(self.scaler.fit_transform(X_train[:, 2:5]), dtype=torch.float32)
        X_test_scaled[:, 2:5] = torch.tensor(self.scaler.transform(X_test[:, 2:5]), dtype=torch.float32)

        self.y_scaler = StandardScaler()
        
        y_train_reshaped = y_train.numpy().reshape(-1, 1)
        y_test_reshaped = y_test.numpy().reshape(-1, 1)
        y_train_scaled = torch.tensor(self.y_scaler.fit_transform(y_train_reshaped), dtype=torch.float32)

        self.model = PyTorchModel(num_unique_buses=num_unique_buses)

        learning_rate = 0.0005
        batch_size = 4096

        self.net = NeuralNet(
            module=self.model,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            lr=learning_rate,
            max_epochs=100,
            batch_size=batch_size,
            train_split=ValidSplit(cv=0.2)
        )

        print('Started training...')
        self.net.fit(X_train_scaled, y_train_scaled)

        losses = self.net.history[:, 'train_loss']

        try:
            z_test_scaled_np = self.net.predict(X_test_scaled)

            z_test_unscaled_np = self.y_scaler.inverse_transform(z_test_scaled_np)
            z_test = torch.tensor(z_test_unscaled_np)

            acc = abs(z_test - y_test.view(-1, 1)).float().mean()
            error = False
        except Exception as e:
            print(f"Error during evaluation: {e}")
            acc = 0
            error = True

        if error:
            print('Error during average error')
        else:
            print(f"\nAverage error: {acc.item():.2f}")
            sample_y_true = y_test.view(-1, 1)
            print(f'test size = {y_test.size(0)}')

        try:
            correct = 0
            wrong = 0
            for x in range(0, sample_y_true.size(0)-1, 10):
                predicted_val = z_test[x].item()
                true_val = sample_y_true[x].item()

                if abs(predicted_val - true_val) < 20:
                    correct += 1
                else:
                    wrong += 1
            print(f'Correct: {(correct / (correct + wrong))*100}%')
            print(f'Learning rate: {learning_rate}, batch_size: {batch_size}')
        except Exception as e:
            print(f'Error calculating accuracy: {e}')


        if graph:
            plt.figure(figsize=(8, 4))
            plt.plot(losses, color='blue', linewidth=2)
            plt.title("Training Loss over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Logistic Loss")
            plt.grid(True, linestyle='--', alpha=0.6)
            save_path = os.path.join(os.path.dirname(__file__), 'training_loss.png')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            print(f'Saved plot to {save_path}')

    def save_weights(self):
        torch.save(self.net.module_.state_dict(), f'./parameters/model_route_{self.route}.pth')
        joblib.dump(self.scaler, f'./parameters/scaler_route_{self.route}.joblib')
        joblib.dump(self.encoder, f'./parameters/encoder_route_{self.route}.joblib')
        joblib.dump(self.y_scaler, f'./parameters/y_scaler_route_{self.route}.joblib')
        print(f'Weights saved for route {self.route}.')

if __name__ == '__main__':
    network = NeuralNetwork('510')
    network.train(graph=True)
    network.save_weights()