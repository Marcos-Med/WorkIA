from losses.loss_crossentropy import LossCrossEntropy

class BackPropagation:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.__loss_fn = LossCrossEntropy()
        self.__lr = learning_rate
        self.__epochs = epochs

    def train(self, layers, X, y):
        for epoch in range(1, self.__epochs+1):
            # Forward pass
            output = X
            for layer in layers:
                output = layer.forward(output)

            # Calcula loss
            loss = self.__loss_fn.forward(output, y)

            # Inicia backward pass com gradiente da loss
            dvalues = self.__loss_fn.backward(output, y)

            # Backprop em cada layer (em ordem reversa)
            for layer in reversed(layers):
                dvalues = layer.backward(dvalues, self.__lr)

            if epoch % 100 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{self.__epochs} - Loss: {loss:.6f}")
