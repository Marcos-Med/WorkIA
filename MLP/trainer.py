from losses.loss_crossentropy import LossCrossEntropy

class BackPropagation:
    def __init__(self, layers, learning_rate=0.01, epochs=1000):
        self.layers = layers
        self.loss_fn = LossCrossEntropy()
        self.lr = learning_rate
        self.epochs = epochs

    def train(self, X, y):
        for epoch in range(1, self.epochs + 1):
            # Forward pass
            output = X
            for layer in self.layers:
                output = layer.forward(output)

            # Calcula loss
            loss = self.loss_fn.forward(output, y)

            # Inicia backward pass com gradiente da loss
            dvalues = self.loss_fn.backward(output, y)

            # Backprop em cada layer (em ordem reversa)
            for layer in reversed(self.layers):
                dvalues = layer.backward(dvalues, self.lr)

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.epochs} - Loss: {loss:.6f}")
