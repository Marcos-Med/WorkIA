import numpy as np

class Sigmoid:
    __instance = None

    def __new__(cls): #Design Patterns Singleton
        if cls.__instance == None:
            cls.__instance = super(Sigmoid, cls).__new__(cls)
        return cls.__instance
    
    """Função de ativação Sigmoid"""
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sig = 1 / (1 + np.exp(-x))
        return sig * (1 - sig)
    
    def dactivation(self, dvalues, z):
        return dvalues * self.derivative(z)
    
    def getName(self): #Nome da função de ativação
        return "Sigmoid"
    