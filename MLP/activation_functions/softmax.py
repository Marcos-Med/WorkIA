# activation_functions/softmax.py
import numpy as np

class Softmax:
    """Função de ativação Softmax"""

    def activation(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # estabilidade numérica
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def derivative(self, x):
        # Nota: a derivada correta do softmax é uma matriz jacobiana
        # Para simplificação no seu projeto, a gente não usa derivative porque o loss já é combinado com softmax no backward
        return np.ones_like(x)  # placeholder
