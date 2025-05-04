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
        self.__inputs = inputs           # armazenar para backprop
        self.__z = np.dot(inputs, self.__weights) + self.__biases
        return self.__activation(self.__z)

    def backward(self, dvalues, learning_rate):
        
        dactivation = self.__activation.dactivation(dvalues, self.__z)

        dweights = np.dot(self.__inputs.T, dactivation)
        dbiases = np.sum(dactivation, axis=0, keepdims=True)
        dinputs = np.dot(dactivation, self.__weights.T)

        self.__weights -= learning_rate * dweights
        self.__biases -= learning_rate * dbiases

        return dinputs
    
    def getNameActivation(self): #Devolve o nome da função ativação
        return self.__activation.getName()
    
    def getQuantityNeurons(self):
        return self.__weights.shape[1] #neurons
    
    def reset(self):
        self.__weights = np.random.randn(*self.__weights.shape) * 0.01