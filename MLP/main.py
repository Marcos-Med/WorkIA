import numpy as np
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data
import nnfs

from mlp import MLP
from trainer import BackPropagation
from layer import Layer
from activation_functions.ReLU import ReLU
from activation_functions.softmax import Softmax

# Inicializa NNFS (fixa semente e formata dados)
nnfs.init()

# 1. Gera os dados
X, y_raw = spiral_data(samples=100, classes=3)

# 2. One-hot encoding dos rótulos
y = np.zeros((len(y_raw), y_raw.max() + 1))
y[np.arange(len(y_raw)), y_raw] = 1

# 3. Define arquitetura: 2 entradas, hidden layer com ReLU, saída com Softmax
configs = [
    (64, 2, ReLU()),        # 2 -> 64
    (3, 64, Softmax())      # 64 -> 3 (classes)
]

# 4. Instancia camadas e trainer
layers = [Layer(activation, n_neurons, n_inputs) for n_neurons, n_inputs, activation in configs]
trainer = BackPropagation(layers, learning_rate=1.0, epochs=100000)
mlp = MLP(layers, trainer)

# 5. Treinamento
mlp.train(X, y)

# 6. Predição
predictions = mlp.predict(X)
predicted_classes = np.argmax(predictions, axis=1)

# 7. Visualização do resultado
plt.scatter(X[:, 0], X[:, 1], c=predicted_classes, cmap='brg')
plt.title("Resultado após treino")
plt.show()
