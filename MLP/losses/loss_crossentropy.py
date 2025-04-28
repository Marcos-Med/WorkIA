# losses/loss_crossentropy.py
import numpy as np

class LossCrossEntropy:
    """Cross-Entropy Loss para classificação"""

    def forward(self, predictions, targets):
        """Calcula o erro"""
        # Clip para evitar log(0)
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        # Se targets estiver one-hot
        if len(targets.shape) == 2:
            correct_confidences = np.sum(predictions_clipped * targets, axis=1)
        else: # targets como rótulos inteiros
            correct_confidences = predictions_clipped[range(len(predictions_clipped)), targets]

        # Negative log likelihood
        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)

    def backward(self, predictions, targets):
        """Calcula o gradiente da Loss"""
        samples = len(predictions)

        # Clip para evitar divisão por 0
        predictions_clipped = np.clip(predictions, 1e-7, 1 - 1e-7)

        if len(targets.shape) == 2:
            # Se targets for one-hot
            dvalues = -targets / predictions_clipped
        else:
            # Se targets forem rótulos inteiros
            dvalues = np.zeros_like(predictions_clipped)
            dvalues[range(samples), targets] = -1 / predictions_clipped[range(samples), targets]

        # Normalizar pelos samples
        return dvalues / samples
