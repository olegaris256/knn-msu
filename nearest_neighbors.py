from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance, cosine_distance
import numpy as np
from scipy.spatial.distance import cdist


class KNNClassifier:
    def __init__(self, k, strategy, metric, weights, test_block_size=1000):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        if self.strategy == 'my_own':
            pass
        elif self.strategy in ['brute', 'ball_tree', 'kd_tree']:
            self.nn_method = NearestNeighbors(n_neighbors=self.k,
                                              algorithm=self.strategy,
                                              metric=self.metric)
            self.nn_method.fit(self.X_train)

    def find_kneighbors(self, X, return_distance):
        distances = np.zeros((X.shape[0], self.k))
        indices = np.zeros((X.shape[0], self.k), dtype=int)
        for start_idx in range(0, X.shape[0], self.test_block_size):
            end_idx = min(start_idx + self.test_block_size, X.shape[0])
            block = X[start_idx:end_idx]
            if self.strategy == 'my_own':
                if self.metric == 'euclidean':
                    distance_matrix = euclidean_distance(block, self.X_train)
                elif self.metric == 'cosine':
                    distance_matrix = cosine_distance(block, self.X_train)
                block_indices = np.argpartition(distance_matrix,
                                                self.k, axis=1)[:, :self.k]
                block_distances = np.take_along_axis(distance_matrix,
                                                     block_indices, axis=1)
            else:
                block_distances, block_indices = self.nn_method.kneighbors(
                    block,
                    n_neighbors=self.k,
                    return_distance=True
                )
            distances[start_idx:end_idx] = block_distances
            indices[start_idx:end_idx] = block_indices
        if return_distance:
            return distances, indices
        else:
            return indices

    def predict(self, X):
        distances, indices = self.find_kneighbors(X, return_distance=True)
        if self.weights:
            weights = 1 / (distances + 1e-5)

        y_pred = np.zeros(X.shape[0], dtype=self.y_train.dtype)
        for i in range(X.shape[0]):
            neighbors = self.y_train[indices[i]]
            classes = np.unique(neighbors)
            if self.weights:
                counts = ((classes[:, None] == neighbors) * weights[i]).sum(
                    axis=1
                )
            else:
                counts = (classes[:, None] == neighbors).sum(axis=1)
            y_pred[i] = classes[counts.argmax()]
        return y_pred

