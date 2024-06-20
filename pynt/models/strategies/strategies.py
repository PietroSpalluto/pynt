from abc import ABC


class LinearRegressionStrategy(ABC):
    def fit(self, X, y, model):
        pass

    def predict(self, X, model):
        pass
