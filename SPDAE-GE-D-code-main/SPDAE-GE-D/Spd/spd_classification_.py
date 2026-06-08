import scipy
import numpy as np
from sklearn import svm
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from Spd import GrassmannKernel
from Spd import GrassmannDistance
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse.linalg import eigsh
from scipy.sparse import eye
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder

from Spd.spd_dist import SPD_Distance


def print_accuracy(name, index, value, flag=True, width=12):
    if flag:
        print("\r" + "{:{width}s}".format("Method", width=width),
              "{:{width}s}".format("Datasets", width=width),
              "{:{width}s}".format("Part", width=width),
              "{:{width}s}".format(index, width=width) + " " * 30)
    print("{:{width}s}".format(name[2], width=width),
          "{:{width}s}".format(name[3], width=width),
          "{:{width}s}".format(name[4], width=width),
          "{:.{width}f}".format(value, width=width - 2))


class SPDKNN:
    def __init__(self, n_neighbors=1, flag=True):
        self.flag = flag
        self.spd_dist = SPD_Distance()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.metric = self.spd_dist.projection_metric_spd
        # self.metric = self.GD.projection_metric

    def fit(self, X, Y=None):
        self.data_train = X
        dist = self.spd_dist.pairwise_dist(X, self.metric)
        self.knn.fit(dist.T, Y)

    def predict(self, data_test):
        dist = self.spd_dist.non_pair_dist(self.data_train, data_test, self.metric)
        self.t_pred = self.knn.predict(dist.T)
        return self.t_pred



class GrassmannKNN:
    def __init__(self, n_neighbors=1, flag=True):
        self.flag = flag
        self.GD = GrassmannDistance()
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.metric = self.GD.projection_metric_AE
        # self.metric = self.GD.projection_metric

    def fit(self, X, Y=None):
        self.data_train = X
        dist = self.GD.pairwise_dist(X, self.metric)
        self.knn.fit(dist.T, Y)

    def predict(self, data_test):
        dist = self.GD.non_pair_dist(self.data_train, data_test, self.metric)
        self.t_pred = self.knn.predict(dist.T)
        return self.t_pred


class GrassmannSVM(svm.SVC):
    def __init__(self, gamma=0.2, flag=True):
        self.gamma = gamma
        self.flag = flag
        self.GK = GrassmannKernel(gamma=gamma)
        kernel = lambda X, Y: self.GK.non_pair_kernel(X, Y, self.GK.gaussian_projection_kernel)
        super().__init__(kernel=kernel)

    def transform(self, X):
        return super().predict(X)


class GrassmannKernelFDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, n_components=20, kernel=None, robustness_offset=1e-8):
        self.n_components = n_components
        self.robustness_offset = robustness_offset
        self.GK = GrassmannKernel()
        self.kernel = self.GK.projection_kernel if kernel is None else kernel

    def fit(self, X, y):
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        y_onehot = OneHotEncoder().fit_transform(self.y_[:, np.newaxis])
        K = self.GK.pairwise_kernel(X, self.GK.projection_kernel)
        m_classes = y_onehot.T @ K / y_onehot.T.sum(1)
        indices = (y_onehot @ np.arange(self.classes_.size)).astype('i')
        N = K @ (K - m_classes[indices])
        N += eye(self.y_.size) * self.robustness_offset
        m_classes_centered = m_classes - K.mean(1)
        M = m_classes_centered.T @ m_classes_centered
        w, self.weights_ = eigsh(M, self.n_components, N, which='LM')
        return self

    def transform(self, X):
        K = self.GK.non_pair_kernel(X, self.X_, self.GK.projection_kernel)
        return K @ self.weights_


class GrassmannKernelRDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    def __init__(self, kernel=None, lmb=0.001):
        self.lmb = lmb
        self.GK = GrassmannKernel()
        self.kernel = self.GK.projection_kernel if kernel is None else kernel

    def fit(self, X, y):
        n = len(X)
        self._X = X
        self._H = np.identity(n) - 1 / n * np.ones(n) @ np.ones(n).T
        self._E = OneHotEncoder().fit_transform(y.reshape(n, 1))
        _, counts = np.unique(y, return_counts=True)
        K = self.GK.pairwise_kernel(X, self.kernel)
        C = self._H @ K @ self._H
        self._Delta = np.linalg.inv(C + self.lmb * np.identity(n))
        A = self._E.T @ C
        B = self._Delta @ self._E
        self._Pi_12 = np.diag(np.sqrt(1.0 / counts))
        P = self._Pi_12 @ A
        Q = B @ self._Pi_12
        R = P @ Q
        V, self._Gamma, self._U = np.linalg.svd(R, full_matrices=False)
        return self

    def transform(self, X):
        _K = self.GK.non_pair_kernel(X, self._X, self.kernel)
        K = _K - np.mean(_K, axis=0)
        C = self._H @ K.T
        T = self._U @ self._Pi_12 @ self._E.T @ self._Delta
        Z = T @ C
        return Z.T


class GrassmannEmbeddingDA:
    def __init__(self, n_components=5, n_neighbors=5, beta=0.001):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.beta = beta
        self.GK = GrassmannKernel()
        self.GD = GrassmannDistance()
        self.kernel = self.GK.projection_kernel
        self.metric = self.GD.projection_metric

    def compute_adjacency_matrix(self, data, target, mode="inner"):
        dist = self.GK.pairwise_kernel(data, self.GK.gaussian_projection_kernel)
        adjacency_matrix = np.zeros_like(dist)
        for i in range(dist.shape[0]):
            if mode == "inner":
                indices = np.where(target == target[i])[0]
                indices = indices[indices != i]
            else:
                indices = np.where(target != target[i])[0]
            dist_per_sample = dist[i, indices]
            sorted_indices = np.argsort(dist_per_sample)
            if len(sorted_indices) > self.n_neighbors:
                nearest_indices = indices[sorted_indices[-self.n_neighbors:]]
            else:
                nearest_indices = indices[sorted_indices]
            adjacency_matrix[i, nearest_indices] = 1
        return adjacency_matrix

    def fit(self, data, target):
        if np.iscomplex(target).any():
            print("target include complex")
        K = self.GK.pairwise_kernel(data, self.kernel)
        self.K_ = K
        dist = self.GD.pairwise_dist(data, self.metric)
        W_inner = self.compute_adjacency_matrix(data, target, "inner")
        W_outer = self.compute_adjacency_matrix(data, target, "outer")
        W_inner = np.maximum(W_inner, np.transpose(W_inner))
        W_outer = np.maximum(W_outer, np.transpose(W_outer))
        D_inner = np.diag(np.sum(W_inner, axis=1))
        D_outer = np.diag(np.sum(W_outer, axis=1))
        L_inner = D_inner - W_inner
        L_outer = D_outer - W_outer
        up = np.dot(np.dot(K, D_inner), np.transpose(K))
        down = np.dot(np.dot(K, (L_outer + self.beta * W_inner)), np.transpose(K))
        eig_values, eig_vectors = scipy.linalg.eig(a=down, b=up)
        sort_index_ = np.argsort(eig_values)[::-1]
        index_ = sort_index_[: self.n_components]
        self.components = eig_vectors[:, index_]
        if np.iscomplexobj(self.components) and np.isclose(self.components.imag, 0).all():
            self.components = self.components.real
        embedding = np.dot(K, self.components)
        return embedding

    def fit_transform(self, data, target):
        self.embedding_ = self.fit(data, target)
        return self.embedding_
