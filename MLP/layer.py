from neuron import Neuron
import numpy as np

#Classe que define as Camadas Escondidas ou de Saída
class Layer:
    def __init__(self, ativation_function, n_neurons=1, n_inputs=1): #1 Neurônio e 1 entrada da camada anterior por padrão
        self.__neurons = self.__create_neurons(ativation_function, n_neurons, n_inputs) # inicializa os neurônios

    def __create_neurons(self, ativation_function, n_neurons, n_inputs): 
        return [Neuron(ativation_function,n_inputs) for _ in range(n_neurons)] #cria um array de neurônios
    
    def response(self, inputs): 
        array = np.asarray(inputs, dtype=float) #converte em array numpy
        return np.array([neuron.response(array) for neuron in self.__neurons], dtype=float) #devolve a resposta da camada