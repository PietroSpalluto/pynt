from pynt.models.base_models import BaseModel
from pynt.optimizers.optimizers import Optimizer, GradientDescent
from pynt.losses.losses import Loss, LinearRegressionLoss


class LinearRegression(BaseModel):
    """
    Linear regression model
    """
    def __init__(self, n_iters: int = 1000, strategy: str = 'numpy', optimizer: Optimizer = GradientDescent(),
                 loss_fn: Loss = LinearRegressionLoss()) -> None:
        super(LinearRegression, self).__init__(strategy)
        self.n_iters = n_iters
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.weights = None
        self.bias = None

    def fit(self, X, y):
        self.strategy.fit(X, y, self)

    def predict(self, X):
        return self.strategy.predict(X, self)
