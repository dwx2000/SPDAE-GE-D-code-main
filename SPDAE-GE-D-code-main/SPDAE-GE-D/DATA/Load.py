################################################################################
# 本文件用于实现加载数据集的标准化
################################################################################
# 导入模块
import json
from pathlib import Path
import numpy as np
import pandas as pd
import medmnist as mm
from sklearn.preprocessing import MinMaxScaler
from .Preprocessing import Pre_Procession as pp
################################################################################
# 加载数据类
class Load_Data():
    # 初始化部分
    def __init__(self, data_name, is_scaler = True):
        self.PATH = "/".join(Path(__file__).parts[0:Path(__file__).parts.index('SPDAE-GE-D') + 1])
        self.data_name = data_name
        self.Select_Class_SPD()
        self.start_text = "当前正在加载" + data_name + "数据集......"
        self.end_text = "\r" + data_name + "数据集加载完毕！" + " " * 10
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_scaler = is_scaler
        self.opt_class = "SPD"



    def Select_Class_SPD(self):
        self.data_path = "_Data.npy"
        self.target_path = "_Target.npy"


    def Loading_SPD(self):
        print(self.start_text, end="")
        Dp = self.PATH + "/DATA_SPD" + "/" + self.data_name + "/" + self.data_name + self.data_path
        Tp = self.PATH + "/DATA_SPD" + "/" + self.data_name + "/" + self.data_name + self.target_path

        data = np.load(Dp)
        target = np.load(Tp)

        print(self.end_text)
        return np.array(data), np.array(target).reshape(-1)
    # 加载MedMNIST数据集
    def Load_MedMNIST(self):
        print(self.start_text, end="")
        info = mm.INFO[self.data_name]
        DataClass = getattr(mm.dataset, info['python_class'])
        train = DataClass(split='train', root=self.data_path)
        test = DataClass(split='test', root=self.data_path)
        val = DataClass(split='val', root=self.data_path)
        data = np.concatenate((train.imgs, test.imgs, val.imgs))
        data = data.reshape((data.shape[0], -1))
        target = np.concatenate((train.labels, test.labels, val.labels))
        if self.data_name == "pathmnist":
            data, target = self.Load_PathMNIST(data, target)
        if self.data_name == "chestmnist":
            data, target = self.Load_ChestMNIST(data, target)
        if self.is_scaler:
            data = self.scaler.fit_transform(data)
        print(self.end_text)
        return data, target.reshape(-1)
    def Load_PathMNIST(self, data, target):
        uc, conut = np.unique(target, return_counts=True)
        data, _, target, _ = pp().uniform_sampling(data, target, train_size=1500, random_state=42)
        return data, target
    def Load_ChestMNIST(self, data, target):
        temp = np.sum(target, axis=1)
        index_0 = temp == 0
        index_1 = temp == 1
        data_0 = data[index_0]
        data_1, target_1 = data[index_1], target[index_1]
        target_0 = np.zeros((len(data_0), 1))
        target_1 = np.array([np.array(i).argmax() for i in target_1]).reshape((-1, 1))
        data = np.concatenate((data_0, data_1), axis=0)
        target = np.concatenate((target_0, target_1), axis=0)
        return data, target
