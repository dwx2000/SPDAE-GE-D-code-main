################################################################################
# 本文件用于实现流形学习算法模型工厂
################################################################################
# 导入模块
import datetime
import platform

import torch
from DATA import Load_Data  # 数据加载模块

from DATA import Pre_Procession as PP  # 数据预处理模块
from method_base.SPD_model import spd_model



################################################################################
# 工厂类：用于生产不同的算法对象
class spd_factory:
    # 初始化部分，用于设定一些基本参数
    def __init__(self, func_name='spdAE-GE-D', data_name='Ballet', return_time=True, train_size=0.1, sec_part='Experiment',
                 sec_num=0, random_state=None):
        # 定义输出分割线长度
        self.xn = 80
        # 输出初始化信息
        print("#" * self.xn)
        print(func_name + "算法性能测试")  # 输出算法名称
        print("*" * self.xn)
        print("性能指标：")
        print("*" * self.xn)
        print("测试日期：", datetime.date.today())  # 打印当前日期
        print("测试时间：", datetime.datetime.now().time().strftime("%H:%M:%S"))  # 打印当前时间
        print("计算机名：", platform.node())  # 打印计算机名称
        print("操作系统：", platform.system())  # 打印操作系统
        print("解 释 器：", platform.python_version())  # 打印Python解释器版本
        print("数 据 集：", data_name)  # 输出数据集名称
        print("算法名称：", func_name)  # 输出使用的算法名称
        print("*" * self.xn)
        # 初始化一些类的成员变量
        self.data_name = data_name  # 数据集名称
        self.func_name = func_name  # 使用的算法名称
        self.random_state = random_state
        self.return_time = return_time  # 是否返回计算时间
        self.train_size = train_size  # 训练集大小比例
        self.sec_part = sec_part  # 实验部分，默认是"Comparatation"
        self.sec_num = sec_num  # 实验编号

    def Product_SPD_SPDAELE_Object(
            self,
            lamda,
            alpha,
            beta,
            n_neighbors=5,
            optimizer='adam',
            method='all',
    ):
        # 加载指定的SPD流形数据集
        self.data, self.target = Load_Data(self.data_name).Loading_SPD()

        # 将数据集划分为训练集和测试集
        train_index, test_index, sampling = self.split_index(
            data=self.data,
            target=self.target,
            train_size=self.train_size,
            random_state=self.random_state
        )

        self.SPD_SPDAE_LE_Object = spd_model(
            lamda=lamda,
            alpha=alpha,
            beta=beta,
            n_neighbors=n_neighbors,
            train_size=self.train_size,
            random_state=self.random_state,
            data_name=self.data_name,
            func_name=self.func_name,
            return_time=self.return_time,
            sec_part=self.sec_part,
            sec_num=self.sec_num,
            optimizer=optimizer,
            method=method,
        )


        setattr(self.SPD_SPDAE_LE_Object, "train_index", train_index)
        setattr(self.SPD_SPDAE_LE_Object, "test_index", test_index)
        setattr(self.SPD_SPDAE_LE_Object, "sampling", sampling)
        setattr(self.SPD_SPDAE_LE_Object, "space", "SPD")
        return self.SPD_SPDAE_LE_Object


    def split_index(self, data, target, train_size, random_state):
        """
        划分训练集和测试集
        :param data:         全体数据
        :param target:       全体标签
        :param train_size:   训练比例
        :param random_state: 随机种子
        :return:
        """
        train_index, test_index = PP().sub_one_sampling_index(data=data, target=target, train_size=train_size, random_state=random_state)
        sampling = 'sub-one'
        return train_index, test_index, sampling
