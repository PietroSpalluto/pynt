from abc import ABC, abstractmethod

import pynt.models.factories.factories as model_factories


class BaseModel(ABC):
    """
    BaseModel is a base class for all models.
    """
    def __init__(self, strategy: str) -> None:
        self.strategy = model_factories.StrategyFactory.get_strategy(self, strategy)

    @abstractmethod
    def fit(self, X, y):
        """
        Fits the model to the training data.
        :param X: training data
        :param y: target data
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predicts the target for the given data.
        :param X: input data
        :return: predicted target
        """
        pass


class ScoringBaseModel(BaseModel):
    """
    BaseModel is a base class for all models which use scoring methods.
    """
    @abstractmethod
    def score(self, X):
        """
        Predicts the score of the target for the given data.
        :param X: input data
        :return: predicted scores
        """
        pass
