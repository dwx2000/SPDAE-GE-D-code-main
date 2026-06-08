import os
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


from Assess.Assessment import K_Nearest_Neighbors
from Assess.Assessment import Support_Vector_Machine
from Draw import Confusion_Matrix
from Draw import Draw_Embedding
from Spd import SPD_Distance



g_distance = SPD_Distance()
draw = Draw_Embedding(dota_size=15)

class spd_Analysis:
    def __init__(self, object, lamda, beta, optimizer, method, save_file=True, print_results=True):

        self.object = object
        self.save_file = save_file
        self.print_results = print_results
        self.knn = K_Nearest_Neighbors()
        self.svm = Support_Vector_Machine()
        self.spd_d = SPD_Distance()
        self.lamda = lamda
        self.beta = beta
        self.optimizer = optimizer
        self.method = method


        self.best_acc = 0.0
        self.best_f1 = 0.0
        self.best_acc_epoch = -1
        self.best_f1_epoch = -1
        self.best_acc_metrics = {}
        self.best_f1_metrics = {}

        # 初始化结果数据框
        self.result = pd.DataFrame(
            columns=[
                "Method", "Datasets",
                "ACC", "PRE", "F1", "REC",
                "time", "train-size", "sampling"],
            index=["Total"])

        if save_file:
            # 设置保存文件路径
            self.xlsx_path = "-".join(self.object.para[0:4]) + '.xlsx'
            self.cmat_path = "-".join(self.object.para[0:4]) + '-Confusion-Matrix'
            # 初始化混淆矩阵绘制对象
            self.cmat = Confusion_Matrix(path="./Figure-SPD-Confusion/" + self.object.data_name,
                                         filename=self.cmat_path)


        self.results_dir = f"./Results_SPD__k=3{self.object.data_name}/"
        os.makedirs(self.results_dir, exist_ok=True)


        self.log_prefix = f"{self.object.data_name}_{self.object.func_name}_L{self.lamda}_G{self.beta}_{self.optimizer}"


        self.csv_path = f"{self.results_dir}{self.log_prefix}_all_epochs.csv"
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w') as f:
                f.write("epoch,ACC,PRE,F1,REC,lambda,gamma,optimizer,method,dataset,train_size,sampling\n")

        self.xn = 80  # 打印行宽
        self.classifer = ["SPDKNN", "GSVM", "GRLGQ"]  #

    def Analysis(self, epoch, classification=True):
        # 创建目录以保存分析结果和图形
        if self.save_file:
            os.makedirs(
                f"./Analysis-SPD/" + self.object.data_name,
                exist_ok=True)
            os.makedirs(
                f"./Figure-visual/" + self.object.data_name,
                exist_ok=True)

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


        if self.object.space == "SPD":
            if classification:
                # 使用K近邻分类器进行预测
                self.knn.KNN_predict_odds_spd(
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
            # 计算所有指标
            accuracy = accuracy_score(self.object.target_test, classification_label)  # 准确率
            precision = precision_score(self.object.target_test, classification_label, average="macro")  # 精确率
            f1 = f1_score(self.object.target_test, classification_label, average="macro")  # F1-score
            recall = recall_score(self.object.target_test, classification_label, average="macro")  # 召回率

            # 更新结果数据框
            self.result["ACC"].Total = accuracy
            self.result["PRE"].Total = precision
            self.result["F1"].Total = f1
            self.result["REC"].Total = recall

            # 当前epoch的完整指标集合
            current_metrics = {
                "ACC": accuracy,
                "PRE": precision,
                "F1": f1,
                "REC": recall,
                "epoch": epoch + 1,
                "lambda": self.lamda,
                "gamma": self.beta,
                "optimizer": self.optimizer,
                "method": self.method,
                "dataset": self.object.data_name,
                "train_size": self.object.train_size,
                "sampling": self.object.sampling
            }

            ######################################################
            ######################################################
            # 1. 将当前epoch结果写入CSV文件
            with open(self.csv_path, 'a') as f:
                f.write(f"{current_metrics['epoch']},{current_metrics['ACC']:.6f},{current_metrics['PRE']:.6f},")
                f.write(f"{current_metrics['F1']:.6f},{current_metrics['REC']:.6f},")
                f.write(f"{current_metrics['lambda']},{current_metrics['gamma']},{current_metrics['optimizer']},")
                f.write(f"{current_metrics['method']},{current_metrics['dataset']},")
                f.write(f"{current_metrics['train_size']},{current_metrics['sampling']}\n")

            # 2. 更新最佳ACC记录
            if accuracy >= self.best_acc:
                self.best_acc_epoch = epoch + 1
                # draw.path = f'./Figure-visual/{self.object.data_name}'
                
                ##################################可视化##################################
                # matrix = g_distance.pairwise_dist(self.object.embedding_test.numpy(), g_distance.gdist)
                # temp = UMAP(n_components=2, metric="precomputed").fit_transform(matrix)
                # if accuracy > self.best_acc:
                #     draw.filename = f"{self.object.data_name}-best-first-{accuracy:.4f}".replace('.', '_')
                # if accuracy == self.best_acc:
                #     draw.filename = f"{self.object.data_name}-best-epoch-{self.best_acc_epoch}-{accuracy:.4f}".replace('.', '_')
                # draw.Draw_embedding(temp, self.object.target_test)
                #####################################################################################################

                self.best_acc = accuracy
                self.best_acc_metrics = current_metrics.copy()
                print(f"<<< 更新最佳ACC: {self.best_acc:.5f} (epoch {self.best_acc_epoch}) >>>")

                # 保存最佳ACC结果到独立文件
                acc_df = pd.DataFrame([{
                    "Metric": "Best_ACC",
                    "Value": self.best_acc,
                    "Epoch": self.best_acc_epoch,
                    "PRE": self.best_acc_metrics["PRE"],
                    "F1": self.best_acc_metrics["F1"],
                    "REC": self.best_acc_metrics["REC"],
                    "Lambda": self.best_acc_metrics["lambda"],
                    "Gamma": self.best_acc_metrics["gamma"],
                    "Optimizer": self.best_acc_metrics["optimizer"],
                    "Method": self.best_acc_metrics["method"],
                    "Dataset": self.best_acc_metrics["dataset"],
                    "TrainSize": self.best_acc_metrics["train_size"],
                    "Sampling": self.best_acc_metrics["sampling"]
                }])
                acc_df.to_excel(
                    f"{self.results_dir}{self.log_prefix}_best_ACC.xlsx",
                    index=False
                )

            # 3. 更新最佳F1记录
            if f1 >= self.best_f1:
                self.best_f1 = f1
                self.best_f1_epoch = epoch + 1
                self.best_f1_metrics = current_metrics.copy()
                print(f"<<< 更新最佳F1: {self.best_f1:.5f} (epoch {self.best_f1_epoch}) >>>")

                # 保存最佳F1结果到独立文件
                f1_df = pd.DataFrame([{
                    "Metric": "Best_F1",
                    "Value": self.best_f1,
                    "Epoch": self.best_f1_epoch,
                    "ACC": self.best_f1_metrics["ACC"],
                    "PRE": self.best_f1_metrics["PRE"],
                    "REC": self.best_f1_metrics["REC"],
                    "Lambda": self.best_f1_metrics["lambda"],
                    "Gamma": self.best_f1_metrics["gamma"],
                    "Optimizer": self.best_f1_metrics["optimizer"],
                    "Method": self.best_f1_metrics["method"],
                    "Dataset": self.best_f1_metrics["dataset"],
                    "TrainSize": self.best_f1_metrics["train_size"],
                    "Sampling": self.best_f1_metrics["sampling"]
                }])
                f1_df.to_excel(
                    f"{self.results_dir}{self.log_prefix}_best_F1.xlsx",
                    index=False
                )

            # 4. 实时写入文本日志文件
            log_path = f"{self.results_dir}{self.log_prefix}_log.txt"
            with open(log_path, 'a') as f:
                f.write(f"Epoch {epoch + 1}:\n")
                f.write(f"  ACC = {accuracy:.5f}, PRE = {precision:.5f}, F1 = {f1:.5f}, REC = {recall:.5f}\n")
                f.write(f"  Lambda = {self.lamda}, Gamma = {self.beta}, Optimizer = {self.optimizer}\n")
                f.write(f"  Current best ACC: {self.best_acc:.5f} (epoch {self.best_acc_epoch})\n")
                f.write(f"  Current best F1: {self.best_f1:.5f} (epoch {self.best_f1_epoch})\n\n")

        # 记录分析所花费的时间
        self.result["time"].Total = 'not yet'

        print("*" * self.xn)

        # 保存分析结果到文件
        # if self.save_file:
        #     self.result.to_excel(
        #         f"./Analysis-{seed}-SPD/" + self.object.data_name + "/" + self.xlsx_path)  # 保存为Excel文件
        #     # 如果类别数量少于等于15，则绘制混淆矩阵
        #     if len(np.unique(self.object.target)) <= 15:
        #         self.cmat.Drawing(self.object.target_test, classification_label)  # 绘制混淆矩阵