from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    @abstractmethod
    def loss(self, y_true, y_pred):
        """
        Optimizer step
        :param y_true: true labels
        :param y_pred: predicted labels
        :return:
        """
        pass

    def grad(self, X, y_true, y_pred):
        """
        Gradients computation
        :param X: input features
        :param y_true: true labels
        :param y_pred: predicted labels
        :return: 
        """


class LinearRegressionLoss(Loss):
    def loss(self, y, y_pred):
        loss = np.mean((y - y_pred) ** 2)

        return loss

    def grad(self, X, y_true, y_pred):
        dw = (1 / X.shape[0]) * np.dot(X.T, y_pred - y_true)
        db = (1 / X.shape[0]) * np.sum(y_pred - y_true)

        return dw, db
