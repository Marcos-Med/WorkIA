from layer import Layer

class MLP:
    def __init__(self, configs, trainer):
        """
        configs: lista de tuplas (n_neurons, n_inputs, activation_function)
        trainer: objeto responsável pelo treinamento (ex: BackPropagation)
        """
        self.__layers = [Layer(activation, n_neurons, n_inputs) for n_neurons, n_inputs, activation in configs]
        self.__trainer = trainer 
        self.__show_banner()

    def __show_banner(self):
        print("\033[95m" + "="*60)
        print("      🧠  Multi-Layer Perceptron Created! 🧠")
        print("\033[94mMulti-Layer Perceptron Configuration")
        print(f"Epochs: {self.__trainer.get_epochs()}")
        print(f"Learning rate: {self.__trainer.get_learning_rate()}")
        print(f"Number of layers: {len(self.__layers)}")
        print("Architecture:")
        for i, layer in enumerate(self.__layers):
            print(f"  Layer {i+1}: {layer.getQuantityNeurons()} neurons - Activation: {layer.getNameActivation()}")
        print("\033[95m" + "="*60)
        print("\033[0m")

    def train(self, X, y, k=None):
        """Treina a rede"""
        self.__trainer.train(self.__layers, X, y, k)

    def predict(self, X):
        """Retorna a saída da rede para X"""
        output = X
        for layer in self.__layers:
            output = layer.forward(output)
        return output
