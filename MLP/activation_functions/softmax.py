# activation_functions/softmax.py
import numpy as np

class Softmax:
    __instance = None

    def __new__(cls): #Design Patterns Singleton
        if cls.__instance is None:
            cls.__instance = super(Softmax, cls).__new__()
        return cls.__instance
    
    """Função de ativação Softmax"""
    def __call__(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # estabilidade numérica
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def derivative(self, x):
        # Nota: a derivada correta do softmax é uma matriz jacobiana
        # Para simplificação no seu projeto, a gente não usa derivative porque o loss já é combinado com softmax no backward
        return np.ones_like(x)  # placeholder
