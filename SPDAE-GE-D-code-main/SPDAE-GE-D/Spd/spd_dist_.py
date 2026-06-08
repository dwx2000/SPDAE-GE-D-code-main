import numpy as np
import torch


# GrassmannDistance类用于计算Grassmann流形上的距离
class GrassmannDistance:
    def __init__(self):
        pass  # 占位符，表示该方法不执行任何操作

    # 计算输入数据集x中所有点对之间的距离矩阵
    # metric为计算距离的度量函数
    def pairwise_dist(self, x, metric):
        n = len(x)  # 获取数据点的数量
        distance = np.zeros((n, n))  # 初始化距离矩阵
        for i in range(n):
            for j in range(i + 1, n):  # 只计算上三角部分，节省计算量
                distance[i, j] = distance[j, i] = metric(x[i], x[j])  # 对称矩阵
        return distance  # 返回距离矩阵

    # 计算x和y两个数据集之间的距离
    # 如果x和y相同，则调用pairwise_dist函数；否则逐点计算x与y的距离
    def non_pair_dist(self, x, y, metric):
        if np.array_equal(x, y):  # 判断x和y是否相等
            return self.pairwise_dist(x, metric)  # 相等则调用pairwise_dist
        m = len(x)  # x的数据点数量
        n = len(y)  # y的数据点数量
        distance = np.zeros((m, n))  # 初始化距离矩阵
        for i in range(m):
            for j in range(n):
                distance[i, j] = metric(x[i], y[j])  # 计算每个点对的距离
        return distance  # 返回x和y的距离矩阵

    # Frobenius范数平方，计算两个Grassmann点a和b之间的距离
    def f_norm_square(self, a, b):
        assert a.shape == b.shape  # 确保a和b形状相同
        if np.array_equal(a, b):  # 如果a和b相等，返回0
            return 0.0
        # 返回Frobenius范数平方除以2
        return np.square(np.linalg.norm(np.dot(a, a.T) - np.dot(b, b.T))) / 2

    def f_norm_square1(self, a, b):
        assert a.shape == b.shape  # 确保a和b形状相同
        loss_init = 0.0
        loss = torch.tensor(loss_init, dtype=torch.double)
        for i in range(b.shape[0]):
            a_sample = a[i]
            b_sample = b[i]
            norm = torch.linalg.norm(torch.matmul(a_sample, a_sample.T) - torch.matmul(b_sample, b_sample.T))
            norm = 0.5 * torch.square(norm)
            loss += norm
        loss_mean = loss / b.shape[0]
        return loss_mean

    # 计算Geodesic距离，使用Frobenius范数除以√2归一化
    def gdist(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        return np.linalg.norm(np.dot(a, a.T) - np.dot(b, b.T)) / np.sqrt(2)

    # 投影矩阵度量，计算两个点之间的投影距离
    def projection_metric(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]  # 计算奇异值（cosθ）
        costheta[costheta >= 1] = 1.0  # 处理数值误差，使cosθ在[0, 1]区间内
        costheta[costheta <= 0] = 0.0
        dist = a.shape[1] - np.sum(np.square(costheta))  # 投影度量计算
        return np.sqrt(dist)

    def projection_metric_AE(self, a, b):
        assert a.shape == b.shape
        if torch.equal(a, b):  # 使用torch.equal来比较PyTorch张量
            return 0.0
        # 将a, b张量转为numpy数组以便使用np.linalg.svd
        costheta = np.linalg.svd(np.dot(a.T.detach().numpy(), b.detach().numpy()))[1]
        costheta[costheta >= 1] = 1.0  # 处理数值误差，使cosθ在[0, 1]区间内
        costheta[costheta <= 0] = 0.0
        dist = a.shape[1] - np.sum(np.square(costheta))  # 投影度量计算
        return np.sqrt(dist)

    # 投影矩阵距离平方
    def projection_metric_square(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        return a.shape[1] - np.square(np.linalg.norm(np.dot(a.T, b)))

    # Binet-Cauchy度量，计算两个Grassmann点的Binet-Cauchy距离
    def binet_cauchy(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = 1 - np.prod(np.square(costheta))  # 计算Binet-Cauchy距离
        return np.sqrt(dist)

    # 最大相关度距离，返回两个Grassmann点之间的最大相关性距离
    def max_correlation(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.max(1 - np.square(costheta))  # 计算最大相关度距离
        return np.sqrt(dist)

    # 最小相关度距离，返回两个Grassmann点之间的最小相关性距离
    def min_correlation(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.min(1 - np.square(costheta))  # 计算最小相关度距离
        return np.sqrt(dist)

    # Chordal距离（Frobenius范数），用于计算Grassmann点之间的弦距离
    def chordal_distance_fro(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.sqrt(np.sum((1 - costheta) / 2)) * 2  # 计算Frobenius范数下的Chordal距离
        return dist

    # Chordal距离（2-范数），基于最小奇异值的弦距离
    def chordal_distance_2m(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.sqrt((1 - np.min(costheta)) / 2) * 2  # 2-范数下的Chordal距离
        return dist

    # 计算Geodesic距离，基于奇异值的弧度距离
    def geodesic_distance(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.sum(np.arccos(costheta) ** 2)  # 计算Geodesic距离
        return dist

    # 计算两个Grassmann点的平均距离
    def mean_distance(self, a, b):
        assert a.shape == b.shape
        if np.array_equal(a, b):
            return 0.0
        costheta = np.linalg.svd(np.dot(a.T, b))[1]
        costheta[costheta >= 1] = 1.0
        costheta[costheta <= 0] = 0.0
        dist = np.sum(1 - costheta ** 2) / a.shape[1]  # 计算平均距离
        return dist
