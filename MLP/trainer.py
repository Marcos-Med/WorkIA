import numpy as np
import random
from losses.loss_crossentropy import LossCrossEntropy

class BackPropagationBase:
    def __init__(self, layers, learning_rate=0.01, epochs=1000):
        self.layers = layers
        self.loss_fn = LossCrossEntropy()
        self.lr = learning_rate
        self.epochs = epochs

    def forward_backward_pass(self, X, y):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        loss = self.loss_fn.forward(output, y)
        dvalues = self.loss_fn.backward(output, y)
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues, self.lr)
        return loss

    # Compara a resposta do modelo com a esperada 
    def evaluate(self, X, y):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        predicted = np.argmax(output, axis=1)
        actual = np.argmax(y, axis=1)
        return np.mean(predicted == actual)

    # Reseta os pesos no caso da validação cruzada
    def reset_layers(self):
        for layer in self.layers:
            if hasattr(layer, 'Reset'):
                layer.Reset()

    # Garante  estratificação dos dados
    def separate_by_class(self, X, y):
        combined = np.concatenate((X, y), axis=1)
        class_data = {}
        for row in combined:
            features, label = row[:-y.shape[1]], row[-y.shape[1]:]
            key = tuple(label)
            class_data.setdefault(key, []).append([features, label])
        return class_data

class BackPropagation(BackPropagationBase):
    def train(self, X, y):
        for epoch in range(1, self.epochs + 1):
            loss = self.forward_backward_pass(X, y)
            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.6f}")

class BackPropagationCV(BackPropagationBase):
    def train(self, X, y, k=5):
        class_data = self.separate_by_class(X, y)
        folds = [[] for _ in range(k)]

        # Distribuir amostras nos folds
        for samples in class_data.values():
            random.shuffle(samples)
            for idx, sample in enumerate(samples):
                folds[idx % k].append(sample)

        for i in range(k):
            valid = folds[i]
            train = sum(folds[:i] + folds[i+1:], [])

            X_train = np.array([s[0] for s in train])
            y_train = np.array([s[1] for s in train])
            X_valid = np.array([s[0] for s in valid])
            y_valid = np.array([s[1] for s in valid])

            for _ in range(self.epochs):
                self.forward_backward_pass(X_train, y_train)

            acc = self.evaluate(X_valid, y_valid)
            print(f"Fold {i+1}/{k} - Acurácia: {acc:.4f}")

            if i != k - 1:
                self.reset_layers()

class BackPropagationES(BackPropagationBase):
    def train(self, X, y):
        class_data = self.separate_by_class(X, y)
        train_data, valid_data = [], []

        for samples in class_data.values():
            random.shuffle(samples)
            n_val = max(1, int(0.1 * len(samples)))
            valid_data += samples[:n_val]
            train_data += samples[n_val:]

        X_train = np.array([s[0] for s in train_data])
        y_train = np.array([s[1] for s in train_data])
        X_valid = np.array([s[0] for s in valid_data])
        y_valid = np.array([s[1] for s in valid_data])

        best_acc = -1
        patience = self.epochs
        epoch = 0

        while patience > 0:
            self.forward_backward_pass(X_train, y_train)
            acc = self.evaluate(X_valid, y_valid)

            if acc > best_acc:
                best_acc = acc
                patience = self.epochs
            else:
                patience -= 1
            epoch += 1

        print(f"Melhor acurácia\nÉpoca {epoch - self.epochs}: {best_acc:.4f}")
