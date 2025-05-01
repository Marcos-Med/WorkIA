import numpy as np
from mlp import MLP
from trainer import BackPropagation
from layer import Layer
from activation_functions.ReLU import ReLU
from activation_functions.softmax import Softmax

# Caminhos para os arquivos
X_txt_path = "../CARACTERES_COMPLETO/X.txt"
Y_txt_path = "../CARACTERES_COMPLETO/Y_letra.txt"

# Carrega X do .txt manualmente, ignorando valores vazios
X = []
with open(X_txt_path, "r") as f:
    for linha in f:
        valores = [val.strip() for val in linha.strip().split(",") if val.strip() != ""]
        if valores:  # ignora linhas totalmente vazias
            X.append([float(v) for v in valores])

X = np.array(X, dtype=np.float32)

# Verifica se todas as linhas têm o mesmo tamanho
comprimentos = [len(linha) for linha in X]
if len(set(comprimentos)) > 1:
    raise ValueError(f"Linhas com tamanhos diferentes detectadas: {set(comprimentos)}")

# Normaliza X (média 0, desvio 1), com segurança contra divisão por zero
X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-7)

# Carrega Y e converte para índices (A=0, ..., Z=25)
with open(Y_txt_path, "r") as f:
    letras = [linha.strip() for linha in f]
letra_to_index = {chr(i + 65): i for i in range(26)}
y_raw = np.array([letra_to_index[l] for l in letras])
n_classes = 26

# One-hot encoding
y = np.zeros((len(y_raw), n_classes))
y[np.arange(len(y_raw)), y_raw] = 1

# Define arquitetura da MLP
n_inputs = X.shape[1]
configs = [
    (64, n_inputs, ReLU()),
    (n_classes, 64, Softmax())
]

# Instancia camadas e trainer
layers = [Layer(activation, n_neurons, n_inputs) for n_neurons, n_inputs, activation in configs]
trainer = BackPropagation(layers, learning_rate=0.05, epochs=1000)
mlp = MLP(layers, trainer)

# Treinamento
mlp.train(X, y)

# Avaliação
predictions = mlp.predict(X)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_raw)
print(f"Acurácia final: {accuracy:.4f}")
