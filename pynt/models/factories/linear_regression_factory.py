from pynt.models.strategies.linear_regression_strategies import NumPyLinearRegressionStrategy, DaskLinearRegressionStrategy


class LinearRegressionStrategyFactory:
    @staticmethod
    def get_strategy(library):
        if library == 'numpy':
            return NumPyLinearRegressionStrategy()
        elif library == 'dask':
            return DaskLinearRegressionStrategy()
        elif library == 'ray':
            raise NotImplementedError("Ray is not implemented yet")
            # return RayStrategy()
        elif library == 'pyspark':
            raise NotImplementedError("PySpark is not implemented yet")
            # return PySparkStrategy()
        else:
            raise ValueError(f"Unknown library: {library}")
