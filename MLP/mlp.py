from layer import Layer
#Classe que define MLP
class MLP:
    #configs = matriz onde cada linha é uma configuração da camada [quantidade_neurônios, quantidade_entradas, função_ativação]
    def __init__(self, configs, trainer):
        self.__layers = self.__create_layers(configs) # inicializa as camadas escondidas e de saída
        self.__trainer = trainer #objeto que treina a MLP

    def __create_layers(self, configs):
        return [Layer(ativation_funct, n_neurons, n_inputs) for n_neurons, n_inputs, ativation_funct in configs] #cria o array de Camadas

    def train(self, x_train, y_train):
        self.__trainer.train(x_train, y_train, self.__layers) #Treina a MLP

    def predict(self, x_test): #Realiza o teste para um dado
        inputs = x_test 
        for layer in self.__layers: #feed-forward
           inputs = layer.response(inputs)
        return inputs