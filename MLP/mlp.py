from layer import Layer

class MLP:
    def __init__(self, layers, trainer):
        """
        configs: lista de tuplas (n_neurons, n_inputs, activation_function)
        trainer: objeto responsável pelo treinamento (ex: BackPropagation)
        """
        self.layers = layers
        self.trainer = trainer 

    def train(self, X, y):
        """Treina a rede"""
        self.trainer.train(X, y)

    def predict(self, X):
        """Retorna a saída da rede para X"""
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output
