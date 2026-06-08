import gc
from time import perf_counter
import numpy as np
import roman
import torch

from scipy.linalg import logm, eigh
from torch import nn

from method_base.SPD_Analysis import spd_Analysis
from method_base.spdnet.optimizers import MixOptimizer
from utils_layer import spd_net_util as util
from scipy.spatial.distance import pdist, squareform


class spd_model(nn.Module):
    def __init__(
            self,
            lamda =1,
            alpha = 1,
            beta = 1,
            n_neighbors=5,
            train_size=5,
            random_state=517,
            data_name='ETH-80',
            func_name='SPDAE-GE',
            sec_part="Experiment",
            sec_num=1,
            optimizer='adam',
            method='all',
            return_time=True,

    ):
        super().__init__()
        # 正交初始化权重矩阵 [5](@ref)
        self.w_enc = nn.Parameter(self.orthogonal_init(400, 100))
        self.w_dec = nn.Parameter(self.orthogonal_init(400, 100).T)
        self.w_enc1 = nn.Parameter(self.orthogonal_init(100, 50))
        self.w_dec1 = nn.Parameter(self.orthogonal_init(100, 50).T)
        ###################
        ###############
        self.lamda = nn.Parameter(torch.tensor(lamda, dtype=torch.double), requires_grad=True)
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.double), requires_grad=True)
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.double), requires_grad=True)

        self.n_neighbors = n_neighbors
        self.train_size = train_size
        self.random_state = random_state
        self.data_name = data_name
        self.func_name = "-".join([func_name, roman.toRoman(1)])
        self.para = [sec_part, str(sec_num), self.func_name, data_name, ""]
        self.sec_num = sec_num
        self.return_time = return_time
        self.optimizer = optimizer
        self.method = method
        self.time = None

    def forward(self, input):
        batch_size = input.shape[0]
        w_1_pc = self.w_enc.contiguous()
        w_1_e = w_1_pc.view([1, w_1_pc.shape[0], w_1_pc.shape[1]])
        w_2_pc = self.w_enc1.contiguous()
        w_2_e = w_2_pc.view([1, w_2_pc.shape[0], w_2_pc.shape[1]])
        w_3_pc = self.w_dec1.contiguous()
        w_3_d = w_3_pc.view([1, w_3_pc.shape[0], w_3_pc.shape[1]])
        w_4_pc = self.w_dec.contiguous()
        w_4_d = w_4_pc.view([1, w_4_pc.shape[0], w_4_pc.shape[1]])
        #编码器部分
        w_tX = torch.matmul(torch.transpose(w_1_e, dim0=1, dim1=2), input)
        w_tXw = torch.matmul(w_tX, w_1_e)
        X_1 = util.rec_mat_v2(w_tXw)
        w_tX = torch.matmul(torch.transpose(w_2_e, dim0=1, dim1=2), X_1)
        w_tXw = torch.matmul(w_tX, w_2_e)
        X_2 = util.rec_mat_v2(w_tXw)
        #解码器部分
        w_tX = torch.matmul(torch.transpose(w_3_d, dim0=1, dim1=2), X_2)
        w_tXw = torch.matmul(w_tX, w_3_d)
        X_3 = util.rec_mat_v2(w_tXw)
        w_tX = torch.matmul(torch.transpose(w_4_d, dim0=1, dim1=2), X_3)
        w_tXw = torch.matmul(w_tX, w_4_d)
        X_4 = util.rec_mat_v2(w_tXw)

        return X_1, X_2, X_4

    def train_(self, num_epochs=100, loss_threshold=1e-4, verbose=True):

        base_lr = 0.005
        loss_weights_lr = 0.005

        loss_params = [self.lamda, self.beta, self.alpha]
        loss_param_ids = {id(p) for p in loss_params}

        net_params = [p for p in self.parameters() if id(p) not in loss_param_ids]

        opt_net = MixOptimizer(net_params, optimizer=torch.optim.SGD, lr=base_lr)

        opt_loss = torch.optim.Adam(loss_params, lr=loss_weights_lr)


        analysis = spd_Analysis(self, self.lamda.item(), self.beta.item(), self.optimizer, self.method)
        losses = []

        data_train = torch.tensor(self.data_train, dtype=torch.double)
        target_train = torch.tensor(self.target_train, dtype=torch.double)
        data_test = torch.tensor(self.data_test, dtype=torch.double)
        target_test = torch.tensor(self.target_test, dtype=torch.double)

        for epoch in range(num_epochs):

            self.eval()
            with torch.no_grad():
                # 测试集降维
                _, X2_te_epoch, _, = self.forward(data_test)

            # 训练模式
            self.train()
            opt_net.zero_grad()
            opt_loss.zero_grad()

            # 前向传播
            embedding_1_train, embedding_2_train ,reconstructed_train = self.forward(data_train)

            # 计算损失
            total_loss = self.compute_total_loss(data_train, reconstructed_train, embedding_1_train, embedding_2_train, target_train)

            # 反向传播
            total_loss.backward()


            opt_net.step()
            opt_loss.step()

            losses.append(total_loss.item())

            self.lamda.data.clamp_(min=1e-4)
            self.beta.data.clamp_(min=1e-4)
            self.alpha.data.clamp_(min=1e-4)

            # 获取当前权重值用于记录
            curr_lamda = self.lamda.item()
            curr_beta = self.beta.item()
            curr_alpha = self.alpha.item()

            if verbose:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item():.6f}')
                # 打印当前权重
                print(f'   [Weights] lamda: {curr_lamda:.4f}, beta: {curr_beta:.4f}, alpha: {curr_alpha:.4f}')

            self.embedding_train = embedding_2_train
            self.embedding_test = X2_te_epoch

            # 更新 analysis 记录
            analysis.lamda = curr_lamda
            analysis.beta = curr_beta
            analysis.Analysis(epoch)
            gc.collect()



    def fit_transform(self, data, target, num_epoch,knn=5):
        self.time_start = perf_counter()

        self.data = data
        self.target = target
        self.split_data()

        W_GE = self.Graphy_weight_matrix(self.data_train, knn)

        self.W = torch.from_numpy(W_GE)

        self.train_( num_epochs=num_epoch, loss_threshold=1e-5)
        self.time_end = perf_counter()

    def split_data(self):
        self.data_train = self.data[self.train_index]
        self.data_test = self.data[self.test_index]
        self.target_train = self.target[self.train_index]
        self.target_test = self.target[self.test_index]


    def reconstruction_loss_spd(self,Y_ori, Y_rec):


        def matrix_log(X):

            L, U = torch.linalg.eigh(X)

            L = L.clamp(min=1e-6)

            log_L = torch.diag_embed(torch.log(L))
            return U @ log_L @ U.transpose(-2, -1)


        log_ori = matrix_log(Y_ori)
        log_rec = matrix_log(Y_rec)


        loss = torch.nn.functional.mse_loss(log_ori, log_rec, reduction='mean')

        return loss

    def class_center_loss_spd(self, embedding, target):


        L, U = torch.linalg.eigh(embedding)
        L = L.clamp(min=1e-6)


        log_L = torch.diag_embed(torch.log(L))
        log_embed = torch.matmul(U, torch.matmul(log_L, U.transpose(-2, -1)))

        vectors = log_embed.view(embedding.size(0), -1)

        unique_classes = torch.unique(target)
        intra_loss = torch.tensor(0.0, device=embedding.device)
        cls_centers = []
        valid_classes = []

        for cls in unique_classes:
            mask = (target == cls)

            if mask.sum() > 0:
                cls_vecs = vectors[mask]


                center = torch.mean(cls_vecs, dim=0)
                cls_centers.append(center)
                valid_classes.append(cls)


                dists_sq = torch.sum((cls_vecs - center) ** 2, dim=1)
                intra_loss += torch.sum(dists_sq)


        intra_loss = intra_loss / embedding.size(0)


        inter_loss = torch.tensor(0.0, device=embedding.device)


        if len(cls_centers) > 1:

            centers_tensor = torch.stack(cls_centers)
            n_cls = len(cls_centers)

            diff = centers_tensor.unsqueeze(1) - centers_tensor.unsqueeze(0)
            center_dists_sq = torch.sum(diff ** 2, dim=-1)

            mask = ~torch.eye(n_cls, dtype=torch.bool, device=embedding.device)
            valid_dists_sq = center_dists_sq[mask]

            inter_loss = torch.sum(1.0 / (valid_dists_sq + 1e-6))

            inter_loss = inter_loss / (n_cls * (n_cls - 1))

        return intra_loss + inter_loss


    def compute_total_loss(self, data, reconstructed,embedding_1,embedding_2, target):
        # 1. 图嵌入损失
        GE_loss = self.lie_lpp_loss_spd(embedding_1)

        # 2. 重构损失
        rec_loss = self.reconstruction_loss_spd(data, reconstructed)

        # 3. 判别类中心对比损失

        D_loss = self.class_center_loss_spd(embedding_2, target)


        # 总损失整合
        total_loss = (
                self.lamda * GE_loss +
                self.beta * rec_loss
                + self.alpha * D_loss
        )

        # 记录各项损失
        self.GE_loss = GE_loss.item()
        self.rec_loss = rec_loss.item()
        self.D_loss = D_loss.item()
        return total_loss


    def lie_lpp_loss_spd(self,spd_output):

        batch_size = spd_output.shape[0]
        W = self.W
        L, U = torch.linalg.eigh(spd_output)

        L = L.clamp(min=1e-6)

        log_L = torch.diag_embed(torch.log(L))

        log_spd = torch.matmul(U, torch.matmul(log_L, U.transpose(-2, -1)))

        vec_log_spd = log_spd.view(batch_size, -1)


        dist_sq = torch.cdist(vec_log_spd, vec_log_spd, p=2) ** 2


        weighted_dist = W * dist_sq

        loss = 0.5 * torch.sum(weighted_dist)

        loss = loss / (batch_size * batch_size)

        return loss

    # 正交初始化方法
    def orthogonal_init(self, size1, size2):
        A = torch.randn(size1, size1)
        AAT = torch.matmul(A, A.T)
        U1, _, _ = torch.svd(AAT)
        W = U1[:, :size2]
        return W.type(torch.float64)


    def Graphy_weight_matrix(self, X, k=5, t=None):

        n_samples, n_rows, n_cols = X.shape
        vec_log_X = np.zeros((n_samples, n_rows * n_cols))

        for i in range(n_samples):
            eigvals, eigvecs = eigh(X[i])
            eigvals = np.maximum(eigvals, 1e-10)
            log_S = eigvecs @ np.diag(np.log(eigvals)) @ eigvecs.T
            vec_log_X[i, :] = log_S.reshape(-1)

        dists = pdist(vec_log_X, 'euclidean')
        dist_matrix = squareform(dists)
        dist_sq_matrix = dist_matrix ** 2
        if t is None:
            t = np.mean(dist_sq_matrix[dist_sq_matrix > 0])
        full_weights = np.exp(-dist_sq_matrix / t)
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            idx = np.argsort(dist_sq_matrix[i])[1:k + 1]
            W[i, idx] = full_weights[i, idx]
            W[idx, i] = full_weights[idx, i]

        return W






