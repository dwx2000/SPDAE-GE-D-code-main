################################################################################
# 本文件用于实现一些数据的预处理操作
################################################################################
# 导入模块
import numpy as np
################################################################################
# 数据预处理类
class Pre_Procession:
    def __init__(self):
        pass

    def uniform_sampling(
            self,
            data,
            target,
            iter1=None,
            iter2=None,
            iter3=None,
            train_size=0.1,
            random_state=None
    ):
        """
        均匀抽样函数
        用于对data, target, iter1, iter2, iter3进行均匀抽样
        :param data: 被采样的数据
        :param target: 被采样的标签
        :param iter1: 被采样的备用变量1
        :param iter2: 被采样的备用变量2
        :param iter3: 被采样的备用变量3
        :param train_size: 训练数据所占的比例
        train_size >= 1: 每类抽取train_size个样本
        train_size < 1: 每类抽取训练样本的比例为train_size
        :param random_state: 随机种子
        :return: 训练数据, 测试数据, 训练标签, 测试标签, ...
        """
        np.random.seed(random_state)
        unique_categories, category_counts = np.unique(target, return_counts=True)
        sampled_indices = []
        if train_size < 1:
            sample_size = np.ceil(train_size * category_counts).astype(np.int)
        else:
            sample_size = int(train_size) * np.ones_like(category_counts)
        for category, ss in zip(unique_categories, sample_size):
            category_indices = np.where(target == category)[0]
            sampled_indices.extend(np.random.choice(category_indices, size=ss, replace=False))
        remaining_indices = np.setdiff1d(np.arange(data.shape[0]), sampled_indices)
        train_data = data[sampled_indices]
        train_target = target[sampled_indices]
        test_data = data[remaining_indices]
        test_target = target[remaining_indices]
        if iter1 is None:
            return train_data, test_data, train_target, test_target
        elif iter2 is None:
            train_iter1 = iter1[sampled_indices]
            test_iter1 = iter1[remaining_indices]
            return train_data, test_data, train_target, test_target, train_iter1, test_iter1
        elif iter3 is None:
            train_iter1 = iter1[sampled_indices]
            test_iter1 = iter1[remaining_indices]
            train_iter2 = iter2[sampled_indices]
            test_iter2 = iter2[remaining_indices]
            return train_data, test_data, train_target, test_target, train_iter1, test_iter1, train_iter2, test_iter2
        else:
            train_iter1 = iter1[sampled_indices]
            test_iter1 = iter1[remaining_indices]
            train_iter2 = iter2[sampled_indices]
            test_iter2 = iter2[remaining_indices]
            train_iter3 = iter3[sampled_indices]
            test_iter3 = iter3[remaining_indices]
            return train_data, test_data, train_target, test_target, train_iter1, test_iter1, train_iter2, test_iter2, train_iter3, test_iter3

    def sub_one_sampling(
            self,
            data,
            target,
            train_size=1,
            random_state=None
    ):
        """
        减一法抽样函数
        用于对每类data抽取train_size个样本逆行测试，其余样本用于训练
        :param data: 被采样的数据
        :param target: 被采样的标签
        :param train_size: 每类样本抽取的个数
        :param random_state: 随机种子
        :return: 训练数据, 测试数据, 训练标签, 测试标签
        """
        np.random.seed(random_state)
        unique_categories, category_counts = np.unique(target, return_counts=True)
        sampled_indices = []
        for category in unique_categories:
            category_indices = np.where(target == category)[0]
            sampled_indices.extend(np.random.choice(category_indices, size=train_size, replace=False))
        remaining_indices = np.setdiff1d(np.arange(data.shape[0]), sampled_indices)
        train_data = data[remaining_indices]
        train_target = target[remaining_indices]
        test_data = data[sampled_indices]
        test_target = target[sampled_indices]
        return train_data, test_data, train_target, test_target

    def uniform_sampling_index(
            self,
            data,
            target,
            train_size=0.1,
            random_state=None
    ):
        """
        均匀抽样函数
        用于对data, target进行均匀抽样，并返回训练数据和测试数据的索引
        :param data: 被采样的数据
        :param target: 被采样的标签
        :param train_size: 训练数据所占的比例
        train_size >= 1: 每类抽取train_size个样本
        train_size < 1: 每类抽取训练样本的比例为train_size
        :param random_state: 随机种子
        :return: 训练数据索引, 测试数据索引
        """
        np.random.seed(random_state)
        unique_categories, category_counts = np.unique(target, return_counts=True)
        sampled_indices = []
        if train_size < 1:
            sample_size = np.ceil(train_size * category_counts).astype(np.int)
        else:
            sample_size = int(train_size) * np.ones_like(category_counts)
        for category, ss in zip(unique_categories, sample_size):
            category_indices = np.where(target == category)[0]
            sampled_indices.extend(np.random.choice(category_indices, size=ss, replace=False))
        remaining_indices = np.setdiff1d(np.arange(data.shape[0]), sampled_indices)
        return sampled_indices, remaining_indices

    def sub_one_sampling_index(
            self,
            data,
            target,
            train_size=1,
            random_state=None
    ):
        """
        减一法抽样函数
        用于对每类data抽取train_size个样本逆行测试，其余样本用于训练，并返回索引
        :param data: 被采样的数据
        :param target: 被采样的标签
        :param train_size: 每类样本抽取的个数
        :param random_state: 随机种子
        :return: 训练数据索引, 测试数据索引
        """
        np.random.seed(random_state)
        unique_categories, category_counts = np.unique(target, return_counts=True)
        sampled_indices = []
        for category in unique_categories:
            category_indices = np.where(target == category)[0]
            sampled_indices.extend(np.random.choice(category_indices, size=train_size, replace=False))
        remaining_indices = np.setdiff1d(np.arange(data.shape[0]), sampled_indices)
        return remaining_indices, sampled_indices

    def select_target(
            self,
            data,
            target,
            selected
    ):
        """
        标签选择函数
        根据标签进行采样
        :param data: 被采样的数据
        :param target: 被采样的标签
        :param selected: 一个集合，包含了从target中选择的标签
        :return: 抽取的数据，抽取的标签
        """
        D = np.zeros((1, data.shape[1]))
        T = np.zeros((1, ))
        for st in selected:
            selected_data = data[target == st]
            selected_target = target[target == st]
            D = np.concatenate((D, selected_data), axis=0)
            T = np.concatenate((T, selected_target), axis=0)
        data = D[1:]
        target = T[1:]
        return data, target

    def target_mirror(
            self,
            target
    ):
        """
        标签映射函数
        当target不连续时，将target映射为连续的标签
        :param target: 标签向量
        :return: 新标签向量，映射关系
        """
        target = np.array(target).flatten()
        target_dict = dict()
        reflect_dict = dict()
        for sort, t in enumerate(np.unique(target)):
            reflect_dict[sort] = t
            target_dict[t] = sort
        new_target = [target_dict.get(t) for t in target]
        return np.array(new_target, dtype=int).flatten(), target_dict

    def add_gaussian_noise(
            self,
            data,
            sigma=0.01,
            clip=(0, 1)
    ):
        """
        高斯噪声函数
        为数据添加高斯噪声 N(0, sigma)
        :param data: 数据
        :param sigma: 高斯噪声强度
        :param clip: 限定的数据的范围
        :return: 含有噪声的数据
        """
        noise = np.random.normal(0, sigma, data.shape)
        data_noisy = data + noise
        data_noisy = np.clip(data_noisy, clip[0], clip[1])
        return data_noisy