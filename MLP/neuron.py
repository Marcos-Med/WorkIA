import numpy as np
from ativation_functions.linear_bipolar import LinearBipolar

class Neuron:
    def __init__(self, ativation_function=LinearBipolar(), n_weights=2):
        self.__weights = np.zeros(n_weights, dtype=float)
        self.__bias = 1.2
        self.__ativation_function = ativation_function
    def response(self, inputs):
        inputs = np.array(inputs, dtype=float)
        linear_combination = self.__weights * inputs
        linear_combination = linear_combination.sum() + self.__bias 
        return self.__ativation_function.ativation(linear_combination)