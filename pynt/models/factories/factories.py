from pynt.models import linear_models
from pynt.models.factories import linear_regression_factory


class StrategyFactory:
    @staticmethod
    def get_strategy(model_class, library):
        if isinstance(model_class, linear_models.LinearRegression):
            return linear_regression_factory.LinearRegressionStrategyFactory.get_strategy(library)
        else:
            raise ValueError("Unsupported model class {}".format(model_class))
