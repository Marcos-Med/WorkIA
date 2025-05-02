import numpy as np
from activation_functions.softmax import Softmax

class Layer:
    def __init__(self, activation_function, n_neurons, n_inputs):
        """
        activation_function: objeto com methods activation(x) e derivative(x)
        n_neurons: número de neurônios na camada
        n_inputs: número de entradas da camada
        """
        self.__activation = activation_function
        # Inicializa pesos (n_inputs x n_neurons) e biases (1 x n_neurons)
        self.__weights = np.random.randn(n_inputs, n_neurons) * 0.01
        self.__biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """Propagação para frente"""
        self.inputs = inputs           # armazenar para backprop
        self.z = np.dot(inputs, self.__weights) + self.__biases
        self.output = self.__activation.activation(self.z)
        return self.output

    def backward(self, dvalues, learning_rate):
        """Backward com cuidado com última camada"""
        # Detecta se é Softmax (não aplica derivative)
        if isinstance(self.__activation, Softmax):
            dactivation = dvalues
        else:
            dactivation = dvalues * self.__activation.derivative(self.z)

        self.dweights = np.dot(self.inputs.T, dactivation)
        self.dbiases = np.sum(dactivation, axis=0, keepdims=True)
        dinputs = np.dot(dactivation, self.__weights.T)

        self.__weights -= learning_rate * self.dweights
        self.__biases -= learning_rate * self.dbiases

        return dinputs
