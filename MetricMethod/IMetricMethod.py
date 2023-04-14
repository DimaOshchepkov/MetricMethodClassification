from ._common_imports import *

__all__ = ['IMetricMethod']

class IMetricMethod():

    __method : IMethodOfGetNeighbours
    __methods_factory : MethodOfGetNeighboursFactory

    __metric : IMetric
    __metric_factory : MetricsFactory

    def __init__(self, metric : str, method : str) -> None:
        
        self.__metric_factory = MetricsFactory()
        self.__metric = self.__metric_factory.get_metrics(metric)

        self.__methods_factory = MethodOfGetNeighboursFactory()
        self.__method = self.__methods_factory.get_method(name_method=method, metric=metric)

    def fit(self, data : pd.Series) -> None:
        pass

    @abstractmethod
    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series) -> any:
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X_test : pd.DataFrame, Y_train : pd.Series) -> np.ndarray:
        raise NotImplementedError()