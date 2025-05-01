import numpy as np
from mlp import MLP
from activation_functions.ReLU import ReLU

# Carregue os dados (exemplo fictício)
# X: shape (n_samples, n_features)
# y: shape (n_samples, n_outputs)
X = np.random.randn(200, 3)
y = np.random.randn(200, 1)

# Definição da rede: duas camadas (3->5->1) usando ReLU
configs = [
    (5, 3, ReLU()),  # Hidden layer: 5 neurônios, 3 inputs,  ativação ReLU
    (1, 5, ReLU()),  # Output layer: 1 neurônio, 5 inputs, ativação ReLU (ou outra dependendo do problema)
]

# Instancia a MLP
mlp = MLP(configs, learning_rate=0.01, epochs=1000)

# Treina
mlp.train(X, y)

# Prediz
pred = mlp.predict(X)
print("Predictions shape:", pred.shape)