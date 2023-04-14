from ._common_imports import *
from .IMetricMethod import *

__all__ = ['KNN']

class KNN(IMetricMethod):

    def __init__(self, metric : str = "euclidean", method : str = "exhaustive") -> None:
        super().__init__(metric, method)

    def __Get_Neighbor(self, train_Y : pd.Series, data_point : pd.Series, countNeighbor : int) -> any:
        ''' Эта функция фозвращает класс соседа объекта data_point, который встречается чаще всего'''

        nearest_index, distances = self.__method.get_neighbours(point=data_point, knn=countNeighbor)
       
        c_neighbor = np.take(train_Y, nearest_index)
        unique, counts = np.unique(c_neighbor, return_counts=True)

        return unique[np.argmax(counts)]  
    
    def fit(self, data : pd.Series) -> None:
        self.__method.preprocessing(data)

    def predict(self, X_test : pd.DataFrame,
                Y_train : pd.Series, count_neigbors : int = 10) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            predict.append(self.__Get_Neighbor(Y_train, row, count_neigbors))

        return np.array(predict) 