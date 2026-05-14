import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNet
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored
import sqlite3
db_conn = sqlite3.connect('tram_data.db', check_same_thread=False)
cursor = db_conn.cursor()

# here data is taken from the SQLite database
X_data = cursor.execute("SELECT * FROM vehicle_locations WHERE datatype = 'VP';").fetchall()
y_data = cursor.execute("SELECT time FROM vehicle_locations WHERE datatype = 'ARS' OR datatype = 'PAS';").fetchall()

X = torch.tensor([[n[2], n[3], n[4]] for n in X_data]) # the input data
y = torch.tensor([n[0] for n in y_data]) # the target
X = X[:len(y_data)] # this is to ensure that the size of X matches y

X_train, X_test = X[:(X.size(0) // 2)], X[(X.size(0) // 2)+1:]
y_train, y_test = y[:len(y_data) // 2], y[(len(y_data) // 2)+1:]
print(f'train sizes: {X_train.size(), y_train.size()}')
print(f'test sizes: {X_test.size(), y_test.size()}')

model = nn.Sequential(
    nn.Linear(3, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
print('Started learning...')
net = NeuralNet(
    module=model,
    criterion=nn.MSELoss,
    optimizer=optim.SGD,
    lr=0.01,
    max_epochs=1000,
    batch_size=1,
    train_split=None
)

net.fit(X_train, y_train)

losses = net.history[:, 'train_loss']

model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)

    z_test = model(X_test_tensor)


    acc = abs(z_test - y_test_tensor).float().mean()

print(f"\nAverage error: {acc.item()}")

plt.figure(figsize=(8, 4))
plt.plot(losses, color='blue', linewidth=2)
plt.title("Training Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Logistic Loss")
plt.grid(True, linestyle='--', alpha=0.6)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Test Set Predictions (10 Samples)", fontsize=16, y=1.05)
axes = axes.flatten()

sample_X = torch.tensor(X_test[:100])
sample_y_true = y_test[:100]

for x in range(100):
    print(f'Batch {x}:')
    print(f'Test result: {sample_X[x]}, correct: {sample_y_true[x]}')
    if abs(z_test - sample_y_true) < 20:
        print(colored('Correct!', 'green'))
    else:
        print(colored('Wrong prediction!', 'red'))
    