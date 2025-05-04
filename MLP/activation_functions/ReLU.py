# activation_functions/relu.py
import numpy as np

class ReLU:
    __instance = None

    def __new__(cls): #Design Patterns Singleton
        if cls.__instance is None: 
            cls.__instance = super(ReLU, cls).__new__(cls)
        return cls.__instance
    
    """Função de ativação ReLU (Rectified Linear Unit)"""
    def __call__(self, x): #ReLU(10) = y
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1.0, 0.0)
    
    def dactivation(self, dvalues, z):
        return dvalues * self.derivative(z)
    
    def getName(self): #Nome da função
        return "ReLU"
