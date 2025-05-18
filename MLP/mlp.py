from .layer import Layer
import json
from .activation_functions.ReLU import ReLU
from .activation_functions.softmax import Softmax
from .activation_functions.tanh import Tanh
from .activation_functions.sigmoid import Sigmoid
from .activation_functions.leakyReLU import LeakyReLU
from .activation_functions.ELU import ELU
from .losses.loss_crossentropy import LossCrossEntropy
from .losses.loss_mse import LossMSE
from .trainer import *

class MLP: #MultiLayer Perceptron
    def __init__(self, configs, trainer):
        """
        configs: lista de tuplas (n_neurons, n_inputs, activation_function)
        trainer: objeto responsável pelo treinamento (ex: BackPropagation)
        """
        self.__layers = [Layer(activation, n_neurons, n_inputs) for n_neurons, n_inputs, activation in configs]
        self.__trainer = trainer 
        self.__show_banner()

    def __show_banner(self):
        print("\033[95m" + "="*60)
        print("      🧠  Multi-Layer Perceptron Created! 🧠")
        print("\033[94mMulti-Layer Perceptron Configuration")
        print(f"Epochs: {self.__trainer.get_epochs()}")
        print(f"Learning rate: {self.__trainer.get_learning_rate()}")
        print(f"Number of layers: {len(self.__layers)}")
        print("Architecture:")
        for i, layer in enumerate(self.__layers):
            print(f"  Layer {i+1}: {layer.getQuantityNeurons()} neurons - Activation: {layer.getNameActivation()}")
        print("\033[95m" + "="*60)
        print("\033[0m")

    def train(self, X, y, k=None):
        """Treina a rede"""
        return self.__trainer.train(self.__layers, X, y, k)

    def predict(self, X):
        """Retorna a saída da rede para X"""
        output = X
        for layer in self.__layers:
            output = layer.forward(output)
        return output
    
    def _setWeights(self, weights, biases): #Coloca os pesos indicados pelo arquivo .model
        i = 0
        for layer in self.__layers:
            layer.setWeights(weights[i])
            layer.setBiases(biases[i])
            i+=1

    def save(self, file_name): #salva o modelo
        model = {
            "weights": [layer.getWeights().tolist() for layer in self.__layers],
            "biases": [layer.getBiases().tolist() for layer in self.__layers],
            "fn_activations": [{"name":layer.getNameActivation(), "params": layer.getParamsActivation()} for layer in self.__layers],
            "training_info": {
                "epochs": self.__trainer.get_epochs(),
                "learning_rate": self.__trainer.get_learning_rate(),
                "loss": self.__trainer.get_lossName(),
                "type": self.__trainer.getType()
            }
        }
        with open(file_name, "w") as file:
            json.dump(model, file, indent=2)

    def load_model(path): #carrega o modelo
            with open(path, "r") as file:
                model = json.load(file)
            activation_factory = {
                "relu": ReLU,
                "sigmoid": Sigmoid,
                "softmax": Softmax,
                "tanh": Tanh,
                "leakyrelu": LeakyReLU,
                "elu": ELU
            }
            loss_factory = {
                "mse": LossMSE,
                "crossentropy": LossCrossEntropy
            }
            trainer_factory = {
                "BackPropagation": BackPropagation,
                "BackPropagationCV": BackPropagationCV,
                "BackPropagationES": BackPropagationES
            }
            configs = []
            for i in range(0, len(model['weights'])): #Adiciona cada configuração das camadas em configs
                if model['fn_activations'][i]["params"] == None: #Caso a função de ativação não tenha hiperparâmetros
                    configs.append((len(model['weights'][i][0]), len(model['weights'][i]), activation_factory[model['fn_activations'][i]['name']]()))
                else:
                    configs.append((len(model['weights'][i][0]), len(model['weights'][i]), activation_factory[model['fn_activations'][i]['name']](**model['fn_activations'][i]['params'])))
            loss = loss_factory[model['training_info']['loss']]()
            trainer = trainer_factory[model['training_info']['type']](loss, model['training_info']['learning_rate'], model['training_info']['epochs'])
            mlp = MLP(configs, trainer)
            mlp._setWeights(model['weights'], model['biases']) #Carrega os pesos
            return mlp
            