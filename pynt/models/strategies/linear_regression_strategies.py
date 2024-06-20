import numpy as np
import dask.array as da
# import ray
# from pyspark.sql import SparkSession

from pynt.models.strategies.strategies import LinearRegressionStrategy
from pynt.parameters.parameters import Parameters


class NumPyLinearRegressionStrategy(LinearRegressionStrategy):
    @staticmethod
    def fit(X, y, model):
        num_samples, num_features = X.shape  # X shape [N, f]
        model.weights = Parameters('weights', np.random.rand(num_features))  # W shape [f, 1]
        model.bias = Parameters('bias', 0)

        for i in range(model.n_iters):
            y_pred = np.dot(X, model.weights.parameters) + model.bias.parameters

            loss = model.loss_fn.loss(y, y_pred)
            model.weights.gradients, model.bias.gradients = model.loss_fn.grad(X, y, y_pred)

            model.optimizer.step({'weights': model.weights, 'bias': model.bias})

    @staticmethod
    def predict(X, model):
        return np.dot(X, model.weights.parameters) + model.bias.parameters


class DaskLinearRegressionStrategy(LinearRegressionStrategy):
    def fit(self, X, y):
        X = da.from_array(X, chunks='auto')
        y = da.from_array(y, chunks='auto')
        num_samples = X.shape[0]
        for _ in range(n_iters):
            y_pred = da.dot(X, weights) + bias
            dw = (1 / num_samples) * da.dot(X.T, y_pred - y).compute()
            db = (1 / num_samples) * da.sum(y_pred - y).compute()
            weights -= lr * dw
            bias -= lr * db
        return weights, bias

    def predict(self, X, weights, bias):
        return np.dot(X, weights) + bias

# @ray.remote
# def compute_gradient(X, y, weights, bias):
#     y_pred = np.dot(X, weights) + bias
#     num_samples = X.shape[0]
#     dw = (1 / num_samples) * np.dot(X.T, y_pred - y)
#     db = (1 / num_samples) * np.sum(y_pred - y)
#     return dw, db
#
# @ray.remote
# def compute_prediction(X, weights, bias):
#     return np.dot(X, weights) + bias
#
# class RayStrategy(Strategy):
#     def fit(self, X, y, weights, bias, lr, n_iters):
#         ray.init(ignore_reinit_error=True)
#         for _ in range(n_iters):
#             dw, db = ray.get(compute_gradient.remote(X, y, weights, bias))
#             weights -= lr * dw
#             bias -= lr * db
#         ray.shutdown()
#         return weights, bias
#
#     def predict(self, X, weights, bias):
#         return ray.get(compute_prediction.remote(X, weights, bias))
#
# class PySparkStrategy(Strategy):
#     def __init__(self):
#         self.spark = SparkSession.builder.appName("LinearRegression").getOrCreate()
#
#     def fit(self, X, y, weights, bias, lr, n_iters):
#         num_samples, num_features = X.shape
#         X_rdd = self.spark.sparkContext.parallelize(X)
#         y_rdd = self.spark.sparkContext.parallelize(y)
#         for _ in range(n_iters):
#             y_pred_rdd = X_rdd.map(lambda row: np.dot(row, weights) + bias)
#             errors_rdd = y_rdd.zip(y_pred_rdd).map(lambda p: p[1] - p[0])
#             dw = np.array(X_rdd.zip(errors_rdd).map(lambda p: p[0] * p[1]).mean()) * (1 / num_samples)
#             db = errors_rdd.mean() * (1 / num_samples)
#             weights -= lr * dw
#             bias -= lr * db
#         return weights, bias
#
#     def predict(self, X, weights, bias):
#         X_rdd = self.spark.sparkContext.parallelize(X)
#         return X_rdd.map(lambda row: np.dot(row, weights) + bias).collect()
#
#     def __del__(self):
#         self.spark.stop()
