import numpy as np
from ativation_functions.linear_bipolar import LinearBipolar

# Classe que define um Neurônio Artificial
class Neuron:
    def __init__(self, ativation_function=LinearBipolar(), n_weights=1): #Função Linear Bipolar e 1 peso por padrão
        self.__weights = np.zeros(n_weights, dtype=float) #pesos
        self.__bias = 1.2 # bias
        self.__ativation_function = ativation_function #função de ativação
    def response(self, inputs): #Método que processa a entrada e fornece a saída do neurônio
        inputs = np.asarray(inputs, dtype=float) #converte em uma array Numpy
        linear_combination = self.__weights * inputs #combinação linear pesos x entradas
        linear_combination = linear_combination.sum() + self.__bias  # soma ao bias
        return self.__ativation_function.ativation(linear_combination) #aplica a função de ativação