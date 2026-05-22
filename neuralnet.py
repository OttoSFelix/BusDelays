import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from skorch.dataset import ValidSplit
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import sqlite3
import math
import os

class NeuralNetwork:
    def __init__(self, route):
        self.db_conn_data = sqlite3.connect('bus_data.db', check_same_thread=False)
        self.cursor_data = self.db_conn_data.cursor()
        self.route = route
        
        self.weights = []
        self.model = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.net = NeuralNet(
            module=self.model,
            criterion=nn.MSELoss,
            optimizer=optim.Adam,
            lr=0.003,
            max_epochs=3000,
            batch_size=256,
            train_split=ValidSplit(cv=5, stratified=False)
        )

    def train(self, graph: bool = False):
        data = self.cursor_data.execute('SELECT * FROM vehicle_locations WHERE route = ? AND delay IS NOT NULL LIMIT 100000;', (self.route,)).fetchall()

        X_data = []
        for n in data:
            time_sec = n[2]
            time_sin = math.sin(time_sec * (2 * math.pi / 86400))
            time_cos = math.cos(time_sec * (2 * math.pi / 86400))
            latitude = n[3]
            longitude = n[4]
            direction = float(n[5])
            X_data.append([time_sin, time_cos, latitude, longitude, direction])

        X = torch.tensor(X_data, dtype=torch.float32)
        y = torch.tensor([n[6] for n in data], dtype=torch.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f'train sizes: {X_train.size(), y_train.size()}')
        print(f'test sizes: {X_test.size(), y_test.size()}')

        scaler = StandardScaler()
        X_train_scaled = X_train.clone()
        X_test_scaled = X_test.clone()
        X_train_scaled[:, 2:4] = torch.tensor(scaler.fit_transform(X_train[:, 2:4]), dtype=torch.float32)
        X_test_scaled[:, 2:4] = torch.tensor(scaler.transform(X_test[:, 2:4]), dtype=torch.float32)

        self.net.fit(X_train_scaled, y_train.view(-1, 1))

        losses = self.net.history[:, 'train_loss']
        evaluations = []

        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

            z_test = self.model(X_test_tensor)

            acc = abs(z_test - y_test.view(-1, 1)).float().mean()

        print(f"\nAverage error: {acc.item():.2f}")
        sample_y_true = y_test.view(-1, 1)
        print(f'test size = {y_test.size(0)}')

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
        torch.save(self.model.state_dict(), f'model_route_{self.route}.pth')

if __name__ == '__main__':
    network = NeuralNetwork('56')
    network.train()