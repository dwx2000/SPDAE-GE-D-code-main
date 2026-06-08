import scipy
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import HDBSCAN
from sklearn.cluster import SpectralClustering
from Spd import GrassmannDistance
from Spd import GrassmannKernel
# https://scikit-learn.org/1.3/modules/classes.html#module-sklearn.cluster
class GrassmannKMeans:
    def __init__(self, center_select="data", eps=10e-6, center_count=5, n_epoch=10, mode="return_label"):
        self.center_select = center_select
        self.eps = eps
        self.distortion_change = []
        self.center_count = center_count
        self.n_epoch = n_epoch
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro
        self.mode = mode
    def recalculate_centers(self, C, X, t):
        U, S, Vh = np.linalg.svd(np.dot(np.dot(np.eye(C.shape[0]) - np.dot(C, C.T), X), np.linalg.pinv(np.dot(C.T, X))), full_matrices=False)
        Y = np.dot(C, np.dot(Vh.T, np.diag(np.cos(np.arctan(S) * t)))) + np.dot(U, np.diag(np.sin(np.arctan(S) * t)))
        Q = np.linalg.qr(Y)[0]
        return Q

    def cluster_distortion(self, dist, labels, center_count):
        cluster_dist = []
        for i in range(center_count):
            idx = (labels == i).nonzero()
            cluster_dist.append(dist[i, idx].mean())
        return np.mean(cluster_dist)

    def fit(self, data):
        if self.center_select.lower() == "data":
            centers = data[np.random.choice(data.shape[0], self.center_count, replace=False)]
        elif self.center_select.lower() == "random":
            centers = []
            for i in range(self.center_count):
                centers.append(np.linalg.qr(np.random.random(data[0].shape))[0])
            centers = np.array(centers)
        else:
            print("Center selection algorithm is invalid.")
            return
        count = 0
        dist = self.GD.non_pair_dist(centers, data, self.metric)
        labels = np.argmin(dist, axis=0)
        avg_dist = self.cluster_distortion(dist, labels, self.center_count)
        self.distortion_change.append(avg_dist)
        delta = 1
        n = np.zeros((1, self.center_count))[0]
        while count < self.n_epoch and delta > self.eps:
            for i in range(data.shape[0]):
                dist = self.GD.non_pair_dist(centers, data[i].reshape((1, data.shape[1], data.shape[2])), self.metric)
                label = np.argmin(dist, axis=0)[0]
                n[label] += 1
                centers[label] = self.recalculate_centers(centers[label], data[i], 1/(n[label]))
            dist = self.GD.non_pair_dist(centers, data, self.metric)
            labels = np.argmin(dist, axis=0)
            avg_dist = self.cluster_distortion(dist, labels, self.center_count)
            delta = (self.distortion_change[-1]-avg_dist)/avg_dist
            self.distortion_change.append(avg_dist)
            count += 1
        dist = self.GD.non_pair_dist(centers, data, self.metric)
        labels = np.argmin(dist, axis=0)
        avg_dist = self.cluster_distortion(dist, labels, self.center_count)
        self.distortion_change.append(avg_dist)
        return centers, labels

    def fit_transform(self, data):
        centers, labels = self.fit(data)
        if self.mode == "return_label":
            return labels
        elif self.mode == "return_center":
            return centers
        elif self.mode == "all":
            return centers, labels

class GrassmannLBG:
    def __init__(
            self,
            center_select="data",
            eps=10e-6,
            center_count=5,
            n_epoch=10,
            mode="return_label"):
        self.center_select = center_select
        self.eps = eps
        self.distortion_change = []
        self.center_count = center_count
        self.n_epoch = n_epoch
        self.GD = GrassmannDistance()
        self.metric = self.GD.chordal_distance_fro
        self.mode = mode

    def flag_mean(self, X, r=None):
        if r is None:
            r = X[0].shape[1]
        A = X[0]
        for i in range(len(X) - 1):
            A = np.hstack((A, X[i + 1]))
        U = np.linalg.svd(A, full_matrices=False)[0]
        return U[:, :r]

    def cluster_distortion(self, dist, labels, center_count):
        cluster_dist = []
        for i in range(center_count):
            idx = (labels == i).nonzero()
            cluster_dist.append(dist[i, idx].mean())
        return np.mean(cluster_dist)
    def init_cemters(self, data):
        if self.center_select.lower() == "data":
            centers = data[np.random.choice(data.shape[0], self.center_count, replace=False)]
        elif self.center_select.lower() == "random":
            centers = []
            for i in range(self.center_count):
                centers.append(np.linalg.qr(np.random.random(data[0].shape))[0])
            centers = np.array(centers)
        else:
            print("Center selection algorithm is invalid.")
            return
        return centers

    def recalculate_label(self, data, centers):
        dist = self.GD.non_pair_dist(centers, data, self.metric)
        labels = np.argmin(dist, axis=0)
        avg_dist = self.cluster_distortion(dist, labels, self.center_count)
        self.distortion_change.append(avg_dist)
        return labels

    def calculate_center(self, data, labels):
        centers = []
        for i in range(self.center_count):
            cluster_subset = data[labels == i]
            if cluster_subset.shape[0] != 0:
                centers.append(self.flag_mean(cluster_subset))
        centers = np.array(centers)
        return centers

    def fit(self, data):
        centers = self.init_cemters(data)
        labels = self.recalculate_label(data, centers)
        count = 0
        delta = 1
        while count < self.n_epoch and delta > self.eps:
            centers = self.calculate_center(data, labels)
            labels = self.recalculate_label(data, centers)
            count += 1
            delta = np.abs(self.distortion_change[-2] - self.distortion_change[-1]) / self.distortion_change[-1]
        return centers, labels

    def fit_transform(self, data):
        flag = True
        while flag:
            try:
                centers, labels = self.fit(data)
                flag = False
            except:
                print("重试！")
                flag = True
        if self.mode == "return_label":
            return labels
        elif self.mode == "return_center":
            return centers
        elif self.mode == "all":
            return centers, labels

class CGMKE:
    def __init__(self, center_count=5):
        self.center_count = center_count
        self.KM = KMeans(n_clusters=center_count)
        self.GK = GrassmannKernel()
        self.kernel = self.GK.projection_kernel

    def top_eigenvectors(self, K, n_components):
        eig_values, eig_vectors = scipy.linalg.eigh(K)
        sort_index_ = np.argsort(eig_values)[::-1]
        index_ = sort_index_[: n_components]
        return eig_vectors[:, index_]

    def trans_data(self, data):
        Kmetrix = self.GK.pairwise_kernel(data, self.kernel)
        D = np.diag(np.power(np.sum(Kmetrix, axis=1), -0.5))
        K = np.dot(np.dot(D, Kmetrix), D)
        n_components = self.center_count - 1 if self.center_count > 2 else self.center_count
        U = self.top_eigenvectors(K, n_components)
        return U

    def fit(self, data):
        kernel_data = self.trans_data(data)
        self.KM.fit(kernel_data)
        return self.KM.labels_

    def fit_transform(self, data):
        labels = self.fit(data)
        return labels
