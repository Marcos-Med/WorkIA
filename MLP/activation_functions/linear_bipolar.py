import numpy as np

class LinearBipolar:
    """Ativação bipolar (-1, +1) com derivada constante"""
    def activation(self, x):
        return np.where(x >= 0, 1, -1)

    def derivative(self, x):
        # aproximação: derivada constante = 1
        return np.ones_like(x)