## This document contains some notes about how different neural network achitectures performed in terms of accuracy (%)

`
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1)
)
`

lr=0.001, epochs=3000, batch_size=32
*Accuracy* **73%**

`
model = nn.Sequential(
    nn.Linear(4, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 16),
    nn.LeakyReLU(),
    nn.Linear(16, 1)
)
`

lr=0.001, epochs=3000, batch_size=32
*Accuracy* **64%**

`
model = nn.Sequential(
    nn.Linear(4, 45),
    nn.ReLU(),
    nn.Linear(45, 45),
    nn.ReLU(),
    nn.Linear(45, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
`

lr=0.001, epochs=3000, batch_size=32
*Accuracy* **72.8%**

`
model = nn.Sequential(
    nn.Linear(4, 45),
    nn.ReLU(),
    nn.Linear(45, 45),
    nn.ReLU(),
    nn.Linear(45, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)
`

lr=0.003, epochs=2500, batch_size=128
*Accuracy* **78.6%**

### From here on, time is splitted into two features: sin(time) and cos(time)

`
model = nn.Sequential(
    nn.Linear(5, 45),
    nn.ReLU(),
    nn.Linear(45, 45),
    nn.ReLU(),
    nn.Linear(45, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.ReLU(),
    nn.Linear(10, 1)
)
`

lr=0.01, epochs=3000, batch_size=128
*Accuracy* **82.7%**

`
model = nn.Sequential(
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
`

lr=0.001, epochs=2000, batch_size=256
*Accurary* **65.4%**

`
model = nn.Sequential(
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
`

lr=0.003, epochs=2000, batch_size=256
*Accuracy* **83.7%**

`
model = nn.Sequential(
    nn.Linear(5, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)
`

lr=0.003, epochs=2000, batch_size=256
*Accuracy* **75.5%**

### From this point forward, the bus id is added in 4 embeddings and the number of epochs is greatly reduced to avoid overfitting

`
self.main_network = nn.Sequential(
    nn.Linear(9, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)
`

lr=0.003, epochs=200, batch_size=512
*Accuracy* **49.7**