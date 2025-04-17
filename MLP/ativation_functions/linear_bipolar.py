class LinearBipolar:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(LinearBipolar, cls).__new__(cls)
        return cls.__instance
    
    def ativation(self, input):
        return 1 if input >= 0 else -1