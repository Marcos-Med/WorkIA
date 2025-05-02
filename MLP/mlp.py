from layer import Layer

class MLP:
    def __init__(self, configs, trainer):
        """
        configs: lista de tuplas (n_neurons, n_inputs, activation_function)
        trainer: objeto responsável pelo treinamento (ex: BackPropagation)
        """
        self.__layers = [Layer(activation, n_neurons, n_inputs) for n_neurons, n_inputs, activation in configs]
        self.__trainer = trainer 

    def train(self, X, y):
        """Treina a rede"""
        self.__trainer.train(self.__layers, X, y)

    def predict(self, X):
        """Retorna a saída da rede para X"""
        output = X
        for layer in self.__layers:
            output = layer.forward(output)
        return output
