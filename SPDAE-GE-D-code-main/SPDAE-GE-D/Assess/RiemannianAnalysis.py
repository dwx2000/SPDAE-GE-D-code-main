################################################################################
# 本文件用于对Riemannian降维算法的评价进行标准化
################################################################################
# 导入模块
import os
import numpy as np
import pandas as pd
import torch
# from umap import UMAP
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from .Assessment import K_Nearest_Neighbors
from .Assessment import Support_Vector_Machine
from Draw import Confusion_Matrix
from Draw import Draw_Embedding
from Spd import GrassmannDistance
from time import perf_counter



################################################################################
# Riemannian降维算法标准化分析过程
class Analysis_Riemannian:
    def __init__(self, object, save_file=True, print_results=True):
        # 初始化分析对象和一些参数
        self.object = object  # Riemannian降维算法对象
        self.save_file = save_file  # 是否保存结果到文件
        self.print_results = print_results  # 是否打印结果
        self.knn = K_Nearest_Neighbors()  # 初始化K近邻分类器
        self.svm = Support_Vector_Machine()  # 初始化支持向量机分类器
        self.GD = GrassmannDistance()  # 初始化Grassmann距离计算
        # 初始化结果数据框
        self.result = pd.DataFrame(
            columns=[
                "Method", "Datasets",
                "ACC", "PRE", "F1", "REC",
                "time", "train-size", "sampling"],
            index=["Total"])  # 只有一行，总结果

        if save_file:
            # 设置保存文件路径
            self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
            self.cmat_path = "-".join(self.object.para[0:4]) + '-Confusion-Matrix'
            # 初始化混淆矩阵绘制对象
            self.cmat = Confusion_Matrix(path=f"./Figure_comparatation-50-/" + self.object.data_name, filename=self.cmat_path)

        self.xn = 80  # 打印行宽
        self.classifer = ["GKNN", "GSVM", "GRLGQ"]  # 支持的分类器列表

    def Analysis(self, classification=True):
        # 创建目录以保存分析结果和图形
        if self.save_file:
            os.makedirs(f"./Analysis_comparatation-50/" + self.object.data_name, exist_ok=True)
            os.makedirs(f"./Figure_comparatation-50/" + self.object.data_name, exist_ok=True)

        # 打印报告的开始信息
        print("*" * self.xn)
        print(self.object.para[2] + "算法在" + self.object.para[3] + "数据集上的降维效果定量评价报告")
        print("*" * self.xn)

        # 填充结果数据框中的方法、数据集、训练大小和采样信息
        self.result["Method"].Total = self.object.func_name
        self.result["Datasets"].Total = self.object.data_name
        self.result["train-size"].Total = self.object.train_size
        self.result["sampling"].Total = self.object.sampling

        # 判断使用的分类器类型
        if self.object.func_name in self.classifer:
            classification_label = self.object.t_pred  # 获取预测标签

        # 处理Grassmann空间
        if self.object.space == "grassmann":
            if classification:
                # 使用K近邻分类器进行预测
                self.knn.KNN_predict_odds_grassmann(
                    self.object.embedding_train, self.object.embedding_test,
                    self.object.target_train, self.object.target_test,
                    self.object.para)
                classification_label = self.knn.t_pred  # 获取分类预测结果

        # 处理欧几里得空间
        elif self.object.space == "euclidean":
            if classification:
                # 使用K近邻分类器进行预测
                self.knn.KNN_predict_odds_splited(
                    self.object.embedding_train, self.object.embedding_test,
                    self.object.target_train, self.object.target_test,
                    self.object.para)
                classification_label = self.knn.t_pred  # 获取分类预测结果

        # 如果进行分类评估
        if classification:
            # 计算各项指标
            self.result["ACC"].Total = accuracy_score(self.object.target_test, classification_label)  # 准确率
            self.result["PRE"].Total = precision_score(self.object.target_test, classification_label, average="macro")  # 精确率
            self.result["F1"].Total = f1_score(self.object.target_test, classification_label, average="macro")  # F1-score
            self.result["REC"].Total = recall_score(self.object.target_test, classification_label, average="macro")  # 召回率

        # 记录分析所花费的时间
        self.result["time"].Total = self.object.time

        print("*" * self.xn)

        # 保存分析结果到文件
        if self.save_file:
            self.result.to_excel(f"./Analysis_comparatation-50/" + self.object.data_name + "/" + self.xlsx_path)  # 保存为Excel文件
            # 如果类别数量少于等于15，则绘制混淆矩阵
            if len(np.unique(self.object.target)) <= 15:
                self.cmat.Drawing(self.object.target_test, classification_label)  # 绘制混淆矩阵
