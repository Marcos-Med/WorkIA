import numpy as np

class LossMSE:
    """Mean Squared Error"""
    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets
        return np.mean((predictions - targets) ** 2)

    def backward(self, predictions, targets):
        # derivada MSE: 2*(pred-target)/n
        samples = targets.shape[0]
        dvalues = 2 * (predictions - targets) / samples
        return dvalues