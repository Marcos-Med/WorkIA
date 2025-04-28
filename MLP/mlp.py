from layer import Layer
from trainer import Trainer

class MLP:
    def __init__(self, configs, learning_rate=0.01, epochs=1000):
        """
        configs: lista de tuplas (n_neurons, n_inputs, activation_function)
        learning_rate: taxa de aprendizado
        epochs: número de épocas
        """
        self.layers = [Layer(activation, n_neurons, n_inputs) for activation, n_neurons, n_inputs  in configs]
        self.trainer = Trainer(self.layers, learning_rate, epochs)

    def train(self, X, y):
        """Treina a rede"""
        self.trainer.train(X, y)

    def predict(self, X):
        """Retorna a saída da rede para X"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output