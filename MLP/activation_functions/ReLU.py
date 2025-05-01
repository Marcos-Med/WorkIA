# activation_functions/relu.py
import numpy as np

class ReLU:
    """Função de ativação ReLU (Rectified Linear Unit)"""
    def activation(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return np.where(x > 0, 1.0, 0.0)
