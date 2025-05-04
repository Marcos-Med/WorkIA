import numpy as np
import random

class BackPropagation:
    def __init__(self, loss, learning_rate=0.01, epochs=1000):
        self.__loss_fn = loss
        self.__lr = learning_rate
        self.__epochs = epochs

    def train(self, layers, X, y):
        for epoch in range(1, self.__epochs+1):
            # Forward pass
            output = X
            for layer in layers:
                output = layer.forward(output)

            # Calcula loss
            loss = self.__loss_fn.forward(output, y)

            # Inicia backward pass com gradiente da loss
            dvalues = self.__loss_fn.backward(output, y)
            
            # Backprop em cada layer (em ordem reversa)
            for layer in reversed(layers):
                dvalues = layer.backward(dvalues, self.__lr)

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.__epochs} - Loss: {loss:.6f}")

    def get_learning_rate(self):
        return self.__lr
    
    def get_epochs(self):
        return self.__epochs
    
    def evaluate(self,layers, inputs, targets):
        output = inputs
        for layer in layers:
            output = layer.forward(output)
        predicted = np.argmax(output, axis=1)
        actual = np.argmax(targets, axis=1)
        return np.mean(predicted == actual)
    # Garante  estratificação dos dados
    def separate_by_class(self, X, y):
        combined = np.concatenate((X, y), axis=1)
        class_data = {}
        for row in combined:
            features, label = row[:-y.shape[1]], row[-y.shape[1]:]
            key = tuple(label)
            class_data.setdefault(key, []).append([features, label])
        return class_data

class BackPropagationCV:
    def __init__(self, loss, learning_rate=0.01, epochs=1000):
        self.__trainer = BackPropagation(loss, learning_rate, epochs)
    
    def evaluate(self,layers, inputs, targets):
        return self.__trainer.evaluate(layers, inputs, targets)
     
    def separate_by_class(self, X, y):
        return self.__trainer.separate_by_class(X,y)

    def get_learning_rate(self):
        return self.__trainer.get_learning_rate()
    
    def get_epochs(self):
        return self.__trainer.get_epochs()

    def train(self,layers, X, y, k=5):
        class_data = self.separate_by_class(X, y)
        folds = [[] for _ in range(k)]

        # Distribuir amostras nos folds
        for samples in class_data.values():
            random.shuffle(samples)
            for idx, sample in enumerate(samples):
                folds[idx % k].append(sample)
        results = []
        for i in range(k):
            valid = folds[i]
            train = sum(folds[:i] + folds[i+1:], [])

            X_train = np.array([s[0] for s in train])
            y_train = np.array([s[1] for s in train])
            X_valid = np.array([s[0] for s in valid])
            y_valid = np.array([s[1] for s in valid])

            for _ in range(self.get_epochs()):
                self.train(layers, X_train, y_train)

            output = X_valid
            for layer in layers:
                output = layer.forward(output)
            results.append({
                'predictions': output,
                'targets': y_valid
            })

            if i != k - 1:
                for layer in layers:
                    layer.reset()
        return results

class BackPropagationES:
    def __init__(self, loss, learning_rate=0.01, epochs=1000):
        self.__trainer = BackPropagation(loss, learning_rate, epochs)

    def evaluate(self,layers, inputs, targets):
        return self.__trainer.evaluate(layers, inputs, targets)
     
    def separate_by_class(self, X, y):
        return self.__trainer.separate_by_class(X,y)

    def get_learning_rate(self):
        return self.__trainer.get_learning_rate()
    
    def get_epochs(self):
        return self.__trainer.get_epochs()
    
    def train(self,layers, X, y, k=None):
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
        patience = self.get_epochs()
        epoch = 0

        while patience > 0:
            self.__trainer.train(layers, X_train, y_train)
            acc = self.evaluate(layers, X_valid, y_valid)

            if acc > best_acc:
                best_acc = acc
                patience = self.get_epochs()
            else:
                patience -= 1
            epoch += 1

        print(f"Melhor acurácia\nÉpoca {epoch - self.get_epochs()}: {best_acc:.4f}")