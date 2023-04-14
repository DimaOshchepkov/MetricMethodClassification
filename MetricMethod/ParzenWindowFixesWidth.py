from ._common_imports import *
from .IMetricMethod import *

__all__ = ['ParzenWindowFixedWidth']

class ParzenWindowFixedWidth(IMetricMethod):

    __kernel : IKernel
    __kernel_factory = KernelFactory()

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive",
                 kernel : str = "default") -> None:
        super().__init__(metric, method)
        self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)

    def __Get_Neighbor(self, train_Y : pd.Series,
                        data_point : pd.Series, width : float) -> any:
        ''' Эта функция фозвращает класс соседа объекта data_point, который встречается чаще всего'''

        nearest_index, distances = self.__method.get_neighbours(point=data_point, width=width)

        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for i in nearest_index:
            nearest[train_Y.iloc[i]] += self.__kernel.kernel_func(distances[i]/width)

        return max(nearest, key=nearest.get)
    
    def fit(self, data : pd.Series) -> None:
        self.__method.preprocessing(data)

    def predict(self, X_test : pd.DataFrame,
                Y_train : pd.Series, width : float = 10) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(Y_train, row, width))

        return np.array(predict) 