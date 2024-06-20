from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    @abstractmethod
    def step(self, model):
        """
        Optimizer step
        :param model:
        :return:
        """
        pass


class GradientDescent(Optimizer):
    def __init__(self, lr: float = 0.01):
        super(GradientDescent, self).__init__(lr)

    def step(self, model_params: dict):
        for params in model_params:
            model_params[params].update(self.lr)
