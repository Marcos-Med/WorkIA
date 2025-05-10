# activation_functions/ELU.py
import numpy as np

class ELU:
    
    def __init__(self, alpha=1.0):
        self.__alpha = alpha
    
    """Função de ativação ELU """
    def __call__(self, x): #ELU(10) = y
        return np.where(x > 0, x, self.__alpha * (np.exp(x) - 1))

    def derivative(self, x):
        return np.where(x > 0, 1.0, self.__alpha * np.exp(x))
    
    def dactivation(self, dvalues, z):
        return dvalues * self.derivative(z)
    
    def getName(self): #Nome da função
        return "ELU"
