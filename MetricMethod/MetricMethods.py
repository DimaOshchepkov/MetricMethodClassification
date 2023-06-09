from abc import abstractmethod
from typing import Callable, Literal

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .Kernels import *
from .MethodsOfGetNeighbours import *
from .Metrics import *

__all__ = [
    'IMetricMethod', 
    'OneNN', 
    'KNN', 
    'ParzenWindowFixedWidth', 
    'ParzenWindowVariableWidth',
    'PotentialFunction',
    'KNN_weight']

class IMetricMethod():
    """Is a generalization for all metric methods"""
    _method : IMethodOfGetNeighbours
    _methods_factory : MethodOfGetNeighboursFactory

    _metric : IMetric
    _metric_factory : MetricsFactory

    _y_train : np.array

    def __init__(self, metric : Literal["euclidean", "cityblock", "cosine"],
                 method : Literal["exhaustive", "kdtree"]) -> None:
        """Common initializer for all metric methods

        Args:
            metric (str): kind of metric. 
            method (str): nearest neighbor method.         
        """
        self._metric_factory = MetricsFactory()
        self._metric = self._metric_factory.get_metrics(metric)

        self._methods_factory = MethodOfGetNeighboursFactory()
        self._method = self._methods_factory.get_method(name_method=method, metric=metric)

    @abstractmethod
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        """Data preprocessing

        Args:
            data (pd.Series): initial data
            y_train (pd.Series): training data class labels
        """
        raise NotImplementedError()

    @abstractmethod
    def __get_neighbor(self, train_Y : pd.Series,
                        nearest_index : np.ndarray,
                        distances : np.ndarray) -> any:
        """
        Returns the class of the nearest neighbor

        Args:
            train_Y (pd.Series): Training sample class labels 
            data_point (pd.Series): The object for which the nearest neighbor is found
            nearest_index (np.ndarray): The index of the nearest neighbor
            distances (np.ndarray): The distances to the nearest neighbor

        Returns:
            any: Returns nearest neighbor
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        """
        Makes a prediction for X_test

        Args:
            X_test (pd.DataFrame): test set

        Returns:
            np.ndarray: Array of predicted class labels
        """
        raise NotImplementedError()
    

class OneNN(IMetricMethod):
    """One nearest neighbor"""
    def __init__(self, metric : Literal["euclidean", "cityblock", "cosine"] = "euclidean",
                 method : Literal["exhaustive", "kdtree"] = "exhaustive") -> None:
        super().__init__(metric, method)


    def __get_neighbor(self, train_Y : pd.Series,
                        nearest_index : np.ndarray,
                        distances : np.ndarray = None) -> any:
        c_neighbor = np.take(train_Y, nearest_index)
        unique, counts = np.unique(c_neighbor, return_counts=True)

        return unique[np.argmax(counts)]
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        data = np.array(data)
        y_train = np.array(y_train)
        
        self._method.preprocessing(data)
        self._y_train = y_train

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        predict = []
        for row in np.array(X_test):
            nearest_index, _ = self._method.get_neighbours(point=row, knn=1)
            predict.append(self.__get_neighbor(self._y_train, nearest_index))

        return np.array(predict)


class KNN(IMetricMethod):
    """K nearest neighbor"""
    __countNeighbor : int

    def __init__(self, metric : Literal["euclidean", "cityblock", "cosine"] = "euclidean",
                 method : Literal["exhaustive", "kdtree"] = "exhaustive",
                 countNeigbor : int = 10) -> None:
        """
        Args:
            metric (str): kind of metric. Defaults to "euclidean". 
            Possible values: "euclidean", "cityblock", "cosine".

            method (str): nearest neighbor method. Defaults to "exhaustive". 
            Possible values: "exhaustive", "kdtree".
            
            countNeigbor (int): number of nearest neighbors. Defaults to 10.
        """
        super().__init__(metric, method)
        self.__countNeighbor = countNeigbor

    def __get_neighbor(self, train_Y : pd.Series, nearest_index : np.ndarray,
                        distances : np.ndarray = None) -> any:   
        c_neighbor = np.take(train_Y, nearest_index)
        unique, counts = np.unique(c_neighbor, return_counts=True)

        return unique[np.argmax(counts)]
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        data = np.array(data)
        y_train = np.array(y_train)

        self._method.preprocessing(data)
        self._y_train = y_train

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        predict = []
        for row in np.array(X_test):
            nearest_index, _ = self._method.get_neighbours(point=row, knn=self.__countNeighbor)
            predict.append(self.__get_neighbor(self._y_train, nearest_index))

        return np.array(predict) 
    
class KNN_weight(IMetricMethod):
    """K nearest neighbor"""
    __countNeighbor : int
    __weight_func : Callable[[float], float]

    def __init__(self, metric : Literal["euclidean", "cityblock", "cosine"] = "euclidean",
                 method : Literal["exhaustive", "kdtree"] = "exhaustive",
                 countNeigbor : int = 10,
                 weight_func : Callable[[float], float] = lambda x : 1/x) -> None:
        """
        Args:
            metric (Literal["euclidean", "cityblock", "cosine"]): kind of metric.
            weight_func (Callable[[float], float]): weight function for that
            get distance from classifier object and neighbor
            method (str): nearest neighbor method. Defaults to "exhaustive". 
            countNeigbor (int): number of nearest neighbors. Defaults to 10.
        """
        super().__init__(metric, method)
        self.__countNeighbor = countNeigbor    
        self.__weight_func = weight_func

    def __get_neighbor(self, train_Y : pd.Series, nearest_index : np.ndarray,
                        distances : np.ndarray = None) -> any:         
        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for ind, dist in zip(nearest_index, distances):
            nearest[train_Y[ind]] += self.__weight_func(dist)

        return max(nearest, key=nearest.get)
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        data = np.array(data)
        y_train = np.array(y_train)

        self._method.preprocessing(data)
        self._y_train = y_train

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            nearest_index, distances = self._method.get_neighbours(point=row, knn=self.__countNeighbor)
            predict.append(self.__get_neighbor(self._y_train, nearest_index, distances=distances))

        return np.array(predict)
    

class ParzenWindowFixedWidth(IMetricMethod):
    """Variable Width Parzen Window"""
    __kernel : IKernel
    __kernel_factory = KernelFactory()

    __width : float

    def __init__(self, metric : Literal["euclidean", "cityblock", "cosine"] = "euclidean",
                 method : Literal["exhaustive", "kdtree"] = "exhaustive",
                 kernel : Literal["rectangular", "gaussian"] = "rectangular",
                 width : float = 10) -> None:
        """
        Args:
            metric (str): kind of metric. Defaults to "euclidean". 
            method (str): nearest neighbor method. Defaults to "exhaustive". 
            kernel (str): kind of kernel. Defaults to "rectangular".

            width (float): window width. Defaults to 10.
        """
        super().__init__(metric, method)
        self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)
        self.__width = width

    def __get_neighbor(self, train_Y : pd.Series,
                           nearest_index : np.ndarray, distances : np.ndarray) -> any:        
        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for ind, dist in zip(nearest_index, distances):
            nearest[train_Y[ind]] += self.__kernel.kernel_func(dist/self.__width)

        return max(nearest, key=nearest.get)
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        data = np.array(data)
        y_train = np.array(y_train)

        self._method.preprocessing(data)
        self._y_train = np.array(y_train)

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            nearest_index, distances = self._method.get_neighbours(point=row, width=self.__width)
            predict.append(self.__get_neighbor(self._y_train, nearest_index, distances))

        return np.array(predict) 
    

class ParzenWindowVariableWidth(IMetricMethod):
    """Fixed Width Parzen Window"""
    __kernel : IKernel
    __kernel_factory = KernelFactory()

    __countNeighbor : int

    def __init__(self, metric : Literal["euclidean", "cityblock", "cosine"] = "euclidean",
                 method : Literal["exhaustive", "kdtree"] = "exhaustive",
                 kernel : Literal["rectangular", "gaussian"] = "rectangular",
                 countNeighbor : int = 10) -> None:
        """
        Args:
            metric (str): kind of metric. Defaults to "euclidean". 
            method (str): nearest neighbor method. Defaults to "exhaustive". 
            kernel (str): kind of kernel. Defaults to "rectangular". 
            countNeigbor (int): number of nearest neighbors. Defaults to 10.
        """        
        super().__init__(metric, method)
        self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)
        self.__countNeighbor = countNeighbor

    def __get_neighbor(self, train_Y : pd.Series, 
                           nearest_index : np.ndarray, distances : np.ndarray) -> any: 

        width = distances[self.__countNeighbor]

        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for ind, dist in zip(nearest_index[:self.__countNeighbor + 1], distances[:self.__countNeighbor + 1]):
            nearest[train_Y[ind]] += self.__kernel.kernel_func(dist/width)

        return max(nearest, key=nearest.get)
    
    def fit(self, data : pd.Series, y_train : pd.Series) -> None:
        data = np.array(data)
        y_train = np.array(y_train)

        self._method.preprocessing(data)
        self._y_train = y_train

    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            nearest_index, distances = self._method.get_neighbours(point=row, knn=self.__countNeighbor + 1)
            predict.append(self.__get_neighbor(self._y_train, nearest_index, distances))

        return np.array(predict) 
    
    
class PotentialFunction(IMetricMethod):
    """Method of potential functions"""
    __kernel : IKernel
    __kernel_factory = KernelFactory()

    __width : int
    __eps : float

    __potentials : np.ndarray
        
    def __init__(self, metric : Literal["euclidean", "cityblock", "cosine"] = "euclidean",
                 method : Literal["exhaustive", "kdtree"] = "exhaustive",
                 kernel : Literal["rectangular", "gaussian"] = "rectangular",
                 width : float = 0.3, eps : float = 0.05) -> None:
        """
        Args:
            metric (str): kind of metric. Defaults to "euclidean".          
            method (str): nearest neighbor method. Defaults to "exhaustive".            
            kernel (str): kind of kernel. Defaults to "rectangular".              
            width (float): window width. Defaults to 0.3.
            eps (float): error tolerance. Defaults to 0.05.
        """        
        super().__init__(metric, method)
        self.__kernel = self.__kernel_factory.get_kernel(name_kernel=kernel)
        self.__width = width
        self.__eps = eps

    def fit(self, data : pd.Series, y_train : pd.Series) -> None: 
        data = np.array(data)
        y_train = np.array(y_train)
        self._method.preprocessing(data)
        self._y_train = y_train
        self.__potentials = np.zeros(data.shape[0])

        # Distance matrix from each vector to each
        dist_matrix = self._metric.get_distance_matrix(data)
        vectors_dist_less_width = {}

        # Loop through each vector and check if its distance falls within the distance range
        for i, dist in enumerate(dist_matrix):
            vector_distance = dist
            # Get the indexes of vectors that fall within the distance range
            included_indexes = np.where((vector_distance > 0) & (vector_distance <= self.__width))[0]
            # Add the included indexes to the dictionary with the vector index as the key
            vectors_dist_less_width[i] = included_indexes

        err = 1.0
        # As long as the number of errors is greater than the specified
        while (err > self.__eps):
            i = 0
            while (i < 1000):
                i += 1
                # Until we get a class mismatch to update the potentials
                rand = np.random.randint(0, len(vectors_dist_less_width))             
                cl = self.__get_neighbor(y_train, vectors_dist_less_width[rand],
                                        dist_matrix[rand, vectors_dist_less_width[rand]])
                if cl != y_train[rand]:
                    self.__potentials[rand] += 1
                    break
                
            # Counting the number of errors
            predict = self.predict(data) 
            err = 1 - accuracy_score(predict, self._y_train)
             
    def __get_neighbor(self, train_Y : pd.Series, nearest_index : np.ndarray, distances : np.ndarray) -> any:       
        nearest = {cl : 0 for cl in np.unique(train_Y)}
        for ind, dist in zip(nearest_index, distances):
            nearest[train_Y[ind]] += self.__kernel.kernel_func(dist/self.__width) * self.__potentials[ind]

        return max(nearest, key=nearest.get)
        
    def predict(self, X_test : pd.DataFrame) -> np.ndarray:
        
        predict = []
        for row in np.array(X_test):
            nearest_index, distances = self._method.get_neighbours(point=row, width=self.__width)
            predict.append(self.__get_neighbor(self._y_train, nearest_index, distances))

        return np.array(predict)
        