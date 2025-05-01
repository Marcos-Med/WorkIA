import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from mlp import MLP
from trainer import BackPropagation
from layer import Layer
from activation_functions.ReLU import ReLU
from activation_functions.softmax import Softmax

# Caminhos para os arquivos
X_txt_path = "../CARACTERES_COMPLETO/X.txt"
Y_txt_path = "../CARACTERES_COMPLETO/Y_letra.txt"

# Carrega X do .txt manualmente
X = []
with open(X_txt_path, "r") as f:
    for linha in f:
        valores = [val.strip() for val in linha.strip().split(",") if val.strip() != ""]
        if valores:
            X.append([float(v) for v in valores])
X = np.array(X, dtype=np.float32)

# Normalização
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

# Split treino/teste
X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
    X, y, y_raw, test_size=0.2, random_state=42, stratify=y_raw
)

# Arquitetura
n_inputs = X.shape[1]
configs = [
    (64, n_inputs, ReLU()),
    (n_classes, 64, Softmax())
]
layers = [Layer(activation, n_neurons, n_inputs) for n_neurons, n_inputs, activation in configs]
trainer = BackPropagation(layers, learning_rate=0.05, epochs=1000)
mlp = MLP(layers, trainer)

# Treinamento
mlp.train(X_train, y_train)

# Avaliação
predictions = mlp.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_raw_test)
print(f"Acurácia no conjunto de teste: {accuracy:.4f}")

# Matriz de confusão
cm = confusion_matrix(y_raw_test, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[chr(i + 65) for i in range(26)])
disp.plot(cmap="viridis", xticks_rotation=45)
plt.title("Matriz de Confusão - Teste")
plt.tight_layout()
plt.show()
