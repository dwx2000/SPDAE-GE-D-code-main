import numpy as np
from Spd import GrassmannDistance
class GrassmannKernel:
    def __init__(self, gamma=0.2):
        self.gamma = gamma
    def pairwise_kernel(self, x, kernel):
        n = len(x)
        Kmatrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                Kmatrix[i, j] = Kmatrix[j, i] = kernel(x[i], x[j])
        return Kmatrix

    def non_pair_kernel(self, x, y, kernel):
        if np.array_equal(x, y):
            return self.pairwise_kernel(x, kernel)
        m = len(x)
        n = len(y)
        Kmatrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                Kmatrix[i, j] = kernel(x[i], y[j])
        return Kmatrix

    def projection_kernel(self, a, b):
        assert a.shape == b.shape
        return np.square(np.linalg.norm(np.dot(a.T, b), ord='fro'))

    def binet_cauchy_kernel(self, a, b):
        assert a.shape == b.shape
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        return np.prod(np.square(costheta))

    def gaussian_projection_kernel(self, a, b):
        assert a.shape == b.shape
        GD = GrassmannDistance()
        distsquare = GD.projection_metric_square(a, b)
        return np.exp(-self.gamma * distsquare)
