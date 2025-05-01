import numpy as np

class Layer:
    def __init__(self, activation_function, n_neurons, n_inputs):
        """
        activation_function: objeto com methods activation(x) e derivative(x)
        n_neurons: número de neurônios na camada
        n_inputs: número de entradas da camada
        """
        self.activation = activation_function
        # Inicializa pesos (n_inputs x n_neurons) e biases (1 x n_neurons)
        self.weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Propagação para frente"""
        self.inputs = inputs           # armazenar para backprop
        self.z = np.dot(inputs, self.weights) + self.biases
        self.output = self.activation.activation(self.z)
        return self.output

    def backward(self, dvalues, learning_rate):
        """Propagação para trás e atualização de parâmetros"""
        # gradiente da ativação
        dactivation = dvalues * self.activation.derivative(self.z)
        # gradientes de pesos e bias
        self.dweights = np.dot(self.inputs.T, dactivation)
        self.dbiases = np.sum(dactivation, axis=0, keepdims=True)
        # gradiente sobre entradas (para próxima camada)
        dinputs = np.dot(dactivation, self.weights.T)
        # atualização
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases
        return dinputs