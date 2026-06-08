################################################################################
# 本文件用于存储流形学习评价指标
################################################################################
# 导入模块
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from scipy.optimize import linear_sum_assignment
from Spd import SPDKNN
from Spd import GrassmannSVM
from Draw import Draw_Line_Chart
################################################################################
# 输出单个指标
def print_accuracy(name, index, value, flag=True, width = 12):
    if flag:
        print("\r"+"{:{width}s}".format("Method", width =width),
              "{:{width}s}".format("Datasets", width =width),
              "{:{width}s}".format("Part", width =width),
              "{:{width}s}".format(index, width =width) + " " * 30)
    print("{:{width}s}".format(name[2], width =width),
          "{:{width}s}".format(name[3], width =width),
          "{:{width}s}".format(name[4], width =width),
          "{:.{width}f}".format(value, width =width-2))
################################################################################
# KNN分类准确率评价
class K_Nearest_Neighbors():
    def __init__(self, neighbors=1, split_size=10, random_seed=517, flag=True):
        self.random_seed = random_seed
        self.split_size = split_size
        self.flag = flag
        self.kf = KFold(n_splits=split_size, shuffle=True)
        self.knn = KNeighborsClassifier(n_neighbors=neighbors, weights='uniform', algorithm='auto')
        self.spd_knn = SPDKNN(n_neighbors=neighbors)
    # 计算KNN分类准确率
    def KNN_predict_odds(self, x, t, name=None):
        self.accuracy = 0.0
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的KNN分类准确率......", end="")
        for train_index, test_index in self.kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            t_train, t_test = t[train_index], t[test_index]
            self.knn.fit(x_train, t_train)
            self.t_pred = self.knn.predict(x_test)
            self.accuracy += accuracy_score(t_test, self.t_pred)
        self.accuracy /= self.split_size
        if name:
            print_accuracy(name, "KNN", value=self.accuracy, flag=self.flag)
    # 实际应用场景下计算KNN分类准确率
    def KNN_predict_odds_splited(self, x_train, x_test, t_train, t_test, name=None):
        self.accuracy = 0.0
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的KNN分类准确率......", end="")
        self.knn.fit(x_train, t_train)
        self.t_pred = self.knn.predict(x_test)
        self.accuracy = accuracy_score(t_test, self.t_pred)
        if name:
            print_accuracy(name, "KNN", value=self.accuracy, flag=self.flag)

    def KNN_predict_odds_grassmann(self, x_train, x_test, t_train, t_test, name=None):
        self.accuracy = 0.0
        if name and self.flag:
            print("当前正在评估" + name[3] + "数据集上的低维SPD数据在KNN分类器上的分类性能......", end="")
        self.spd_knn.fit(x_train, t_train)
        self.t_pred = self.spd_knn.predict(x_test)
        self.accuracy = accuracy_score(t_test, self.t_pred)
        if name:
            print_accuracy(name, "GKNN", value=self.accuracy, flag=self.flag)

    def KNN_predict_odds_spd(self, x_train, x_test, t_train, t_test, name=None):
        self.accuracy = 0.0
        if name and self.flag:
            print("当前正在评估" + name[3] + "数据集上的低维SPD数据在KNN分类器上的分类性能......", end="")
        self.spd_knn.fit(x_train, t_train)
        self.t_pred = self.spd_knn.predict(x_test)
        self.accuracy = accuracy_score(t_test, self.t_pred)
        if name:
            print_accuracy(name, "SPDKNN", value=self.accuracy, flag=self.flag)

################################################################################
# SVM分类准确率评价
class Support_Vector_Machine():
    def __init__(self, gamma="scale", split_size=10, random_seed=517, flag=True):
        self.random_seed = random_seed
        self.split_size = split_size
        self.flag = flag
        self.kf = KFold(n_splits=split_size, shuffle=True)
        self.svm = SVC(gamma=gamma)
        self.gsvm = GrassmannSVM()
    # 计算SVM分类准确率
    def SVM_predict_odds(self, x, t, name=None):
        self.accuracy = 0.0
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的SVM分类准确率......", end="")
        for train_index, test_index in self.kf.split(x):
            x_train, x_test = x[train_index], x[test_index]
            t_train, t_test = t[train_index], t[test_index]
            self.svm.fit(x_train, t_train)
            t_pred = self.svm.predict(x_test)
            self.accuracy += accuracy_score(t_test, t_pred)
        self.accuracy /= self.split_size
        if name:
            print_accuracy(name, "SVM", value=self.accuracy, flag=self.flag)
    # 实际应用场景下计算SVM分类准确率
    def SVM_predict_odds_splited(self, x_train, x_test, t_train, t_test, name=None):
        self.accuracy = 0.0
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的SVM分类准确率......", end="")
        self.svm.fit(x_train, t_train)
        t_pred = self.svm.predict(x_test)
        self.accuracy = accuracy_score(t_test, t_pred)
        if name:
            print_accuracy(name, "SVM", value=self.accuracy, flag=self.flag)

    def SVM_predict_odds_grassmann(self, x_train, x_test, t_train, t_test, name=None):
        self.accuracy = 0.0
        if name and self.flag:
            print("当前正在评估" + name[3] + "数据集上的低维Grassmann数据在SVM分类器上的分类性能......", end="")
        try:
            self.gsvm.fit(x_train, t_train)
            t_pred = self.gsvm.predict(x_test)
            self.accuracy = accuracy_score(t_test, t_pred)
        except:
            self.accuracy = -1.0
        if name:
            print_accuracy(name, "GSVM", value=self.accuracy, flag=self.flag)

################################################################################
# Adjusted Rand Score指数
class Adjusted_Rand_Score():
    # 初始化部分
    def __init__(self, flag=True, split_size=10, is_split=False):
        self.flag = flag
        self.is_split = is_split
        if is_split:
            self.split_size = split_size
            self.kf = KFold(n_splits=split_size, shuffle=True)
    # 计算Adjusted Rand Score指数
    def adjusted_rand_score_(self, t_true, t_test, name=None):
        self.score = 0.0
        if name and self.flag:
            print("当前正在计算" + name[3] + "数据集上投影的Adjusted Rand Score指数......", end="")
        if self.is_split:
            for train_index, test_index in self.kf.split(t_true):
                true_train, true_test = t_true[train_index], t_true[test_index]
                test_train, test_test = t_test[train_index], t_test[test_index]
                self.score += adjusted_rand_score(true_test, test_test)
            self.score /= self.split_size
        else:
            self.score = adjusted_rand_score(t_true, t_test)
        if name:
            print_accuracy(name, "ARI", value=self.score, flag=self.flag)


################################################################################
# 调整的互信息指数
class Ajusted_Mutual_Info_Score():
    # 初始化部分
    def __init__(self, flag=True, split_size=10, is_split=False):
        self.flag = flag
        self.is_split = is_split
        if is_split:
            self.split_size = split_size
            self.kf = KFold(n_splits=split_size, shuffle=True)
    # 计算调整的互信息指数
    def adjusted_mutual_info_score_(self, t_true, t_test, name=None):
        self.score = 0.0
        if name and self.flag:
            print("当前正在计算" + name[3] + "数据集上投影的调整的互信息指数......", end="")
        if self.is_split:
            for train_index, test_index in self.kf.split(t_true):
                true_train, true_test = t_true[train_index], t_true[test_index]
                test_train, test_test = t_test[train_index], t_test[test_index]
                self.score += adjusted_mutual_info_score(true_test, test_test)
            self.score /= self.split_size
        else:
            self.score = adjusted_mutual_info_score(t_true, t_test)
        if name:
            print_accuracy(name, "AMI", value=self.score, flag=self.flag)

################################################################################
# 标准化互信息指数
class Normalized_Mutual_Info_Score():
    # 初始化部分
    def __init__(self, flag=True, split_size=10, is_split=False):
        self.flag = flag
        self.is_split = is_split
        if is_split:
            self.split_size = split_size
            self.kf = KFold(n_splits=split_size, shuffle=True)
    # 计算标准化互信息指数
    def normalized_mutual_info_score_(self, t_true, t_test, name=None):
        self.score = 0.0
        if name and self.flag:
            print("当前正在计算" + name[3] + "数据集上投影的标准化互信息指数......", end="")
        if self.is_split:
            for train_index, test_index in self.kf.split(t_true):
                true_train, true_test = t_true[train_index], t_true[test_index]
                test_train, test_test = t_test[train_index], t_test[test_index]
                self.score += normalized_mutual_info_score(true_test, test_test)
            self.score /= self.split_size
        else:
            self.score = normalized_mutual_info_score(t_true, t_test)
        if name:
            print_accuracy(name, "NMI", value=self.score, flag=self.flag)

################################################################################
# Homogeneity Score指数
class Homogeneity_Score():
    # 初始化部分
    def __init__(self, flag=True, split_size=10, is_split=False):
        self.flag = flag
        self.is_split = is_split
        if is_split:
            self.split_size = split_size
            self.kf = KFold(n_splits=split_size, shuffle=True)
    # 计算Homogeneity Score指数
    def homogeneity_score_(self, t_true, t_test, name=None):
        self.score = 0.0
        if name and self.flag:
            print("当前正在计算" + name[3] + "数据集上投影的Homogeneity Score指数......", end="")
        if self.is_split:
            for train_index, test_index in self.kf.split(t_true):
                true_train, true_test = t_true[train_index], t_true[test_index]
                test_train, test_test = t_test[train_index], t_test[test_index]
                self.score += homogeneity_score(true_test, test_test)
            self.score /= self.split_size
        else:
            self.score = homogeneity_score(t_true, t_test)
        if name:
            print_accuracy(name, "HMS", value=self.score, flag=self.flag)


################################################################################
# Completeness Score指数
class Completeness_Score():
    # 初始化部分
    def __init__(self, flag=True, split_size=10, is_split=False):
        self.flag = flag
        self.is_split = is_split
        if is_split:
            self.split_size = split_size
            self.kf = KFold(n_splits=split_size, shuffle=True)
    # 计算Completeness Score指数
    def completeness_score_(self, t_true, t_test, name=None):
        self.score = 0.0
        if name and self.flag:
            print("当前正在计算" + name[3] + "数据集上投影的Completeness Score指数......", end="")
        if self.is_split:
            for train_index, test_index in self.kf.split(t_true):
                true_train, true_test = t_true[train_index], t_true[test_index]
                test_train, test_test = t_test[train_index], t_test[test_index]
                self.score += completeness_score(true_test, test_test)
            self.score /= self.split_size
        else:
            self.score = completeness_score(t_true, t_test)
        if name:
            print_accuracy(name, "CMS", value=self.score, flag=self.flag)

################################################################################
# V Measure Score指数
class V_Measure_Score():
    # 初始化部分
    def __init__(self, flag=True, split_size=10, is_split=False):
        self.flag = flag
        self.is_split = is_split
        if is_split:
            self.split_size = split_size
            self.kf = KFold(n_splits=split_size, shuffle=True)
    # 计算V Measure Score指数
    def v_measure_score_(self, t_true, t_test, name=None):
        self.score = 0.0
        if name and self.flag:
            print("当前正在计算" + name[3] + "数据集上投影的V Measure Score指数......", end="")
        if self.is_split:
            for train_index, test_index in self.kf.split(t_true):
                true_train, true_test = t_true[train_index], t_true[test_index]
                test_train, test_test = t_test[train_index], t_test[test_index]
                self.score += v_measure_score(true_test, test_test)
            self.score /= self.split_size
        else:
            self.score = v_measure_score(t_true, t_test)
        if name:
            print_accuracy(name, "VMS", value=self.score, flag=self.flag)

################################################################################
# Fowlkes Mallows指数
class Fowlkes_Mallows_Score():
    # 初始化部分
    def __init__(self, flag=True, split_size=10, is_split=False):
        self.flag = flag
        self.is_split = is_split
        if is_split:
            self.split_size = split_size
            self.kf = KFold(n_splits=split_size, shuffle=True)
    # 计算Fowlkes_Mallows指数
    def fowlkes_mallows_score_(self, t_true, t_test, name=None):
        self.score = 0.0
        if name and self.flag:
            print("当前正在计算" + name[3] + "数据集上投影的Fowlkes Mallows指数......", end="")
        if self.is_split:
            for train_index, test_index in self.kf.split(t_true):
                true_train, true_test = t_true[train_index], t_true[test_index]
                test_train, test_test = t_test[train_index], t_test[test_index]
                self.score += fowlkes_mallows_score(true_test, test_test)
            self.score /= self.split_size
        else:
            self.score = fowlkes_mallows_score(t_true, t_test)
        if name:
            print_accuracy(name, "FMI", value=self.score, flag=self.flag)

################################################################################
# 轮廓系数评价
class Silhouette_Score():
    # 初始化部分
    def __init__(self, flag=True, split_size=10, is_split=False):
        self.flag = flag
        self.is_split = is_split
        if is_split:
            self.split_size = split_size
            self.kf = KFold(n_splits=split_size, shuffle=True)

    # 计算轮廓系数
    def silhouette_score_(self, x, t, name=None):
        self.score = 0.0
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的轮廓系数......", end="")
        if self.is_split:
            for train_index, test_index in self.kf.split(x):
                x_train, x_test = x[train_index], x[test_index]
                t_train, t_test = t[train_index], t[test_index]
                self.score += silhouette_score(x_test, t_test)
            self.score /= self.split_size
        else:
            self.score = silhouette_score(x, t)
        if name:
            print_accuracy(name, "SIL", value=self.score, flag=self.flag)

################################################################################
# Clustering Accuracy
class Cluster_Accuracy:
    def __init__(self, flag = True):
        self.flag = flag

    def find_intersect(self, ip, t_pred, true_class_i, tupel):
        pred_class_i = tupel[t_pred == ip]
        n_intersect = len(np.intersect1d(true_class_i, pred_class_i))
        return ((len(true_class_i) - n_intersect) + (len(pred_class_i) - n_intersect))

    def min_weight_bipartite_matching(self, t_true, t_pred):
        assert t_true.shape == t_pred.shape
        idx_true = np.unique(t_true)
        idx_pred = np.unique(t_pred)
        tupel = np.arange(len(t_true))
        assignmentMatrix = -1 * np.ones((len(idx_true), len(idx_pred)))
        for it in idx_true:
            true_class_i = tupel[t_true == it]
            assignmentMatrix[it] = [self.find_intersect(ip, t_pred, true_class_i, tupel) for ip in idx_pred]
        row_idx, col_idx = linear_sum_assignment(assignmentMatrix)
        return row_idx, col_idx, assignmentMatrix

    def best_map(self, t_true, t_pred):
        t_true, t_pred = t_true.reshape(-1), t_pred.reshape(-1)
        row_ind, col_ind, matrix = self.min_weight_bipartite_matching(t_true, t_pred)
        new_pred = np.zeros_like(t_pred)
        for r, c in zip(row_ind, col_ind):
            new_pred[t_pred == c] = r
        return new_pred.astype(int)
    def calculate_accuracy(self, t_true, t_pred, name = None):
        self.accuracy = 0.0
        if name and self.flag:
            print("当前正在计算" + name[3] + "数据集上投影的Cluster Accuracy......", end="")
        pro_pred = self.best_map(t_true, t_pred)
        self.accuracy = accuracy_score(t_true.flatten(), pro_pred.flatten())
        if name:
            print_accuracy(name, "ACC", value=self.accuracy, flag=self.flag)
        return self.accuracy

################################################################################
# AIC信息准则
class Akaike_Information_Criterion():
    def __init__(self, func_name='', data_names=[], odds=[],
                 max_dimension=[],
                 cut_dimension=[], filename = 'default'):
        self.func_name = func_name
        self.data_names = data_names
        self.odds = odds
        self.max_dimension = max_dimension
        self.cut_dimension = cut_dimension
        self.filename = filename

    def AIC_Run(self):
        self.AIC_Calculate()
        self.AIC_Draw_Pic()
        self.AIC_Print()
    def AIC_Print(self):
        with open('AIC-Best-Dimensionality.txt', 'a') as file:
            for d, ds in enumerate(self.data_names):
                txt = self.func_name + "算法在" + ds + "数据集上的最优降维维度是：" + str(self.aic_1[d]) + "\n"
                print(txt)
                file.write(txt)
            for d, ds in enumerate(self.data_names):
                txt = self.func_name + "算法在" + ds + "数据集上的次优降维维度是：" + str(self.aic_2[d]) + "\n"
                print(txt)
                file.write(txt)
            for d, ds in enumerate(self.data_names):
                txt = self.func_name + "算法在" + ds + "数据集上的再优降维维度是：" + str(self.aic_3[d]) + "\n"
                print(txt)
                file.write(txt)
            mid = np.int64(np.floor(np.mean((self.aic_1, self.aic_2, self.aic_3))))
            txt = self.func_name + "算法在" + "、".join(self.data_names) + "等数据集上的平均降维维度是：" + str(mid) + "\n"
            print(txt)
            file.write(txt)
        file.close()
    def AIC_Draw_Pic(self):
        LC = Draw_Line_Chart(left=self.aic,
            left_label=self.data_names,
            xlabel='Dimensions', ylabel_left='Values of AIC',
            ylim_left=(0, 2),
            left_color=["#b8474d", "#4e6691"],
            filename=self.filename)
        LC.Draw_simple_line()
    def AIC_Calculate(self):
        self.data_num = len(self.data_names)
        self.aic = np.zeros_like(self.odds)
        self.aic_1 = np.zeros_like(np.array(self.max_dimension))
        self.aic_2 = np.zeros_like(np.array(self.max_dimension))
        self.aic_3 = np.zeros_like(np.array(self.max_dimension))
        for i in range(self.data_num):
            for j in range(self.cut_dimension[i]):
                self.aic[i][j] = 1.0-self.odds[i][j]+(j+1)/self.max_dimension[i]
            self.aic_1[i] = np.argsort(self.aic[i])[0] + 1
            self.aic_2[i] = np.argsort(self.aic[i])[1] + 1
            self.aic_3[i] = np.argsort(self.aic[i])[2] + 1

################################################################################
# 联合排名矩阵评价
class Q_matrix():
    # 初始化部分
    def __init__(self, batch_size=100, cross_time=10, flag=True):
        self.batch_size = batch_size
        self.cross_time = cross_time
        self.flag = flag
        self.index_max = None
        self.Kmax = None
        self.Q_local = None
        self.Q_global = None
        self.Qm = None
        self.Bm = None
        self.x_distance = None
        self.y_distance = None

    def Init_matrix(self):
        self.HDS = np.zeros([self.index_max, self.index_max], dtype=np.int64)
        self.LDS = np.zeros([self.index_max, self.index_max], dtype=np.int64)
        self.R = np.zeros([self.index_max, self.index_max], dtype=np.int64)
        self.Q = np.zeros([self.index_max, self.index_max], dtype=np.int64)

    # 计算距离矩阵
    def Calculate_dist(self, x, y):
        self.index_max = x.shape[0]
        self.x_distance = pairwise_distances(x)
        self.y_distance = pairwise_distances(y)

    # 计算联合排名矩阵
    def Calculate_co_ranking(self):
        for i in range(self.index_max):
            self.HDS[i] = self.x_distance[i].argsort()
            self.LDS[i] = self.y_distance[i].argsort()
        self.HDS = self.HDS.T
        self.LDS = self.LDS.T
        for j in range(self.index_max):
            for i in range(self.index_max):
                self.R[self.LDS[i, j], j] = i
        for j in range(self.index_max):
            for i in range(self.index_max):
                k = self.R[self.HDS[i, j], j]
                self.Q[i, k] = self.Q[i, k] + 1
        self.Q = self.Q[1:, 1:]

    # 初始化计算
    def Calculate_Init(self, x, y):
        self.Calculate_dist(x, y)
        self.Init_matrix()
        self.Calculate_co_ranking()
        self.Calculate_Qm()
        self.Calculate_Bm()
        self.Calculate_Kmax()

    # 基础运算
    def Calculate_Q_NX(self, k):
        num = np.sum(self.Q[:k, :k])
        return num / k / self.index_max

    def Calculate_B_NX(self, k):
        B = self.Q[:k, :k]
        num = np.sum(np.triu(B, k=1) - np.tril(B, k=-1))
        return num / k / self.index_max

    # 重要参数运算
    def Calculate_LCMC(self):
        return np.array([self.Qm[k - 1] - k / (self.index_max - 1) for k in range(1, self.index_max)])

    def Calculate_Kmax(self):
        LCMC_K = self.Calculate_LCMC()
        self.Kmax = LCMC_K.argsort()[-1] + 1

    # 质量列表
    def Calculate_Qm(self):
        self.Qm = np.array([self.Calculate_Q_NX(i) for i in range(1, self.index_max)])
        return self.Qm

    def Calculate_Bm(self):
        self.Bm = np.array([self.Calculate_B_NX(i) for i in range(1, self.index_max)])
        return self.Bm

    # 评价指标单次运算
    def Calculate_Q_avg(self):
        self.Q_avg = np.mean(self.Qm)

    def Calculate_B_avg(self):
        self.B_avg = np.mean(self.Bm)

    def Calculate_L(self):
        self.L = (self.index_max - self.Kmax) / (self.index_max - 1)

    def Calculate_Q_local(self):
        self.Q_local = np.sum(self.Qm[:self.Kmax]) / self.Kmax

    def Calculate_Q_global(self):
        self.Q_global = np.sum(self.Qm[self.Kmax:]) / (self.index_max - self.Kmax)

    def Calculate_Total(self):
        self.Calculate_Q_avg()
        self.Calculate_B_avg()
        self.Calculate_L()
        self.Calculate_Q_local()
        self.Calculate_Q_global()

    # 汇总部分
    def Statistic_Qm(self, X, Y):
        if X.shape[0]<self.batch_size:
            self.batch_size = X.shape[0] - 1
            self.cross_time = 1
        Qm_temp = np.zeros(self.batch_size - 1)
        for i in range(self.cross_time):
            x, _, y, _ = train_test_split(X, Y, train_size=self.batch_size)
            self.Calculate_Init(x, y)
            Qm_temp += self.Qm
        self.Statistic_Qm_ = Qm_temp / self.cross_time
        self.Statistic_Qm0 = np.array([self.Statistic_Qm_[i] for i in range(10, self.batch_size - 1, 10)])

    def Statistic_Bm(self, X, Y):
        if X.shape[0]<self.batch_size:
            self.batch_size = X.shape[0] - 1
            self.cross_time = 1
        Bm_temp = np.zeros(self.batch_size - 1)
        for i in range(self.cross_time):
            x, _, y, _ = train_test_split(X, Y, train_size=self.batch_size)
            self.Calculate_Init(x, y)
            Bm_temp += self.Bm
        self.Statistic_Bm_ = Bm_temp / self.cross_time
        self.Statistic_Bm0 = np.array([self.Statistic_Bm_[i] for i in range(10, self.batch_size - 1, 10)])

    def Statistic_Qavg(self, X, Y, name=None):
        self.Statistic_Qm(X, Y)
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的Q_avg指标......", end="")
        self.Statistic_Qmavg = np.mean(self.Statistic_Qm_)
        if name:
            print_accuracy(name, "Q-avg", value=self.Statistic_Qmavg, flag=self.flag)

    def Statistic_Bavg(self, X, Y, name=None):
        self.Statistic_Bm(X, Y)
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的B_avg指标......", end="")
        self.Statistic_Bmavg = np.mean(self.Statistic_Bm_)
        if name:
            print_accuracy(name, "B-avg", value=self.Statistic_Bmavg, flag=self.flag)

    def Statistic_L(self, X, Y, name=None):
        L_temp = 0
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的L指标......", end="")
        if X.shape[0]<self.batch_size:
            self.batch_size = X.shape[0] - 1
            self.cross_time = 1
        for j in range(self.cross_time):
            x, _, y, _ = train_test_split(X, Y, train_size=self.batch_size)
            self.Calculate_Init(x, y)
            self.Calculate_L()
            L_temp += self.L
        self.Statistic_L_ = L_temp / self.cross_time
        if name:
            print_accuracy(name, "L", value=self.Statistic_L_, flag=self.flag)

    def Statistic_Qlocal(self, X, Y, name=None):
        Qlocal_temp = 0
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的Q_local指标......", end="")
        if X.shape[0]<self.batch_size:
            self.batch_size = X.shape[0] - 1
            self.cross_time = 1
        for j in range(self.cross_time):
            x, _, y, _ = train_test_split(X, Y, train_size=self.batch_size)
            self.Calculate_Init(x, y)
            self.Calculate_Q_local()
            Qlocal_temp += self.Q_local
        self.Statistic_Q_local = Qlocal_temp / self.cross_time
        if name:
            print_accuracy(name, "Q-local", value=self.Statistic_Q_local, flag=self.flag)

    def Statistic_Qglobal(self, X, Y, name=None):
        Qglobal_temp = 0
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上投影的Q_global指标......", end="")
        if X.shape[0]<self.batch_size:
            self.batch_size = X.shape[0] - 1
            self.cross_time = 1
        for j in range(self.cross_time):
            x, _, y, _ = train_test_split(X, Y, train_size=self.batch_size)
            self.Calculate_Init(x, y)
            self.Calculate_Q_global()
            Qglobal_temp += self.Q_global
        self.Statistic_Q_global = Qglobal_temp / self.cross_time
        if name:
            print_accuracy(name, "Q-global", value=self.Statistic_Q_global, flag=self.flag)

    def Statistic_Total(self, X, Y, name=None):
        if name and self.flag:
            print("当前正在计算" + name[2] + "算法在" + name[3] + "数据集上联合排名矩阵评价结果......", end="")
        self.Statistic_Qavg(X, Y, name=None)
        self.Statistic_Bavg(X, Y, name=None)
        self.Statistic_L(X, Y, name=None)
        self.Statistic_Qlocal(X, Y, name=None)
        self.Statistic_Qglobal(X, Y, name=None)
