import json
from pathlib import Path
import numpy as np
import torch



class spd_config:
    def __init__(self):
        # 数据集名称列表，表示当前支持的数据集
        self.data_name = [

        ]

        # 用户自己的数据集，初始化为空列表
        self.self_data = []


        # 算法使用的数据集列表
        self.spd_data = [

        ]

        # 未使用的数据集，初始化为空
        self.none_data = []
        # 最近邻的数量，默认为5
        self.n_neighbors = 5
        # 随机种子，基于2025年，保证实验的可重复性
        self.random_state = np.random.randint(2025)

        # 是否返回时间，默认返回
        self.return_time = True

        # 训练集大小比例，每个数据集有不同的划分方式
        self.train_size = {
            "Ballet": 1,
            "ETH-80": 1,
            "Traffic": 5,
            "UCF-S": 1,
            "UT-Kinect": 1

        }



