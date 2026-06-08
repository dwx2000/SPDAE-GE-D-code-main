################################################################################
# 本文件用于绘图函数的标准化
################################################################################
# 导入模块
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
rcParams['figure.autolayout'] = True
DPI=800
PIC_FORMAT=('pdf', 'png')
################################################################################
# 保存函数
def Save_Pictures(name="default", formats=PIC_FORMAT, dpi = DPI):
    for fmt in formats:
        plt.savefig(name+"."+fmt, format=fmt, dpi=dpi, bbox_inches='tight')
    plt.clf()

def Save_Pictures_new(name="default", formats=PIC_FORMAT, dpi=DPI):
    plt.savefig(name + "." + 'png', format='png', dpi=dpi, bbox_inches='tight')
    plt.clf()

################################################################################
# 嵌入效果图绘制
class Draw_Embedding():
    def __init__(self,
                 path='./Figure/',
                 cmap='Spectral',
                 dota_size=2,
                 fontsize=10,
                 titlefontsize=12,
                 filename='default_name',
                 title=None,
                 show_legend = True,
                 lgd=None):
        self.path = path
        self.cmap = cmap
        self.size = dota_size
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.filename = filename
        self.title = title
        self.show_legend = show_legend
        self.lgd = lgd
        os.makedirs(self.path, exist_ok=True)

    def Draw_embedding(self, Embedding, Target, name=None):
        if name is not None:
            topic = "-".join(name)
        else:
            topic = self.filename
        unique_categories, category_counts = np.unique(Target, return_counts=True)
        if self.lgd is not None:
            labels = self.lgd
        else:
            labels = [f"Class {int(i)}" for i in unique_categories]

        sc = plt.scatter(Embedding[:, 0], Embedding[:, 1], c=Target.astype(int), s=self.size,cmap=self.cmap)
        if self.show_legend:

            plt.legend(handles=sc.legend_elements(num=len(unique_categories)-1)[0], loc='upper right', labels=labels, ncol=np.ceil(len(unique_categories)/10), fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)

        Save_Pictures_new(os.path.join(self.path, topic))
    def Draw_embedding_3D(self, Embedding, Target, name=None):
        if name is not None:
            topic = "-".join(name)
        else:
            topic = self.filename
        unique_categories, category_counts = np.unique(Target, return_counts=True)
        if self.lgd is not None:
            labels = self.lgd
        else:
            labels = [f"Class {i}" for i in unique_categories]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(Embedding[:, 0], Embedding[:, 1], Embedding[:, 2], c=Target.astype(int), marker='o')
        if self.show_legend:
            plt.legend(handles=sc.legend_elements(num=len(category_counts))[0], ncol=np.ceil(len(category_counts) / 10), loc='upper right', labels=labels, fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, topic))
################################################################################
# 距离分布对比图绘制
class Draw_Pairhist():
    def __init__(self,
                 bar_num=30,
                 data_name = '',
                 fontsize=8,
                 titlefontsize=12,
                 filename='default',
                 path='./Figure/'):
        self.bar_num = bar_num
        self.data_name = data_name
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.filename = filename
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def format_func(self, value, tick_number):
        return "{:,.0e}".format(abs(value))
    def Draw_pairhist(self, dist_train, dist_test):
        self.range_min = np.min(np.array([np.min(dist_train), np.min(dist_test)]))
        self.range_max = np.max(np.array([np.max(dist_train), np.max(dist_test)]))
        plt.figure()
        hist_up, bins_up = np.histogram(dist_train.ravel(), bins=self.bar_num, range=(self.range_min, self.range_max))
        bin_centers_up = (bins_up[:-1] + bins_up[1:]) / 2
        plt.bar(bin_centers_up, hist_up, width=(bins_up[1] - bins_up[0]), color="blue", edgecolor="white", label='train-data')
        hist_down, bins_down = np.histogram(dist_test.ravel(), bins=self.bar_num, range=(self.range_min, self.range_max))
        bin_centers_down = (bins_down[:-1] + bins_down[1:]) / 2
        plt.bar(bin_centers_down, -hist_down, width=(bins_down[1] - bins_down[0]), color="red", edgecolor="white", label='oos-data')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(self.format_func))
        plt.legend(fontsize=self.fontsize)
        plt.xlabel('Distance values', fontsize=self.titlefontsize)
        plt.ylabel('Distance values in that block', fontsize=self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
# 核方法流形学习训练数据集划分比例分析图绘制
class Draw_Split_Analysis():
    def __init__(
            self, object,
            filename='default',
            path='./Figure/',
            xlabel='% size of train set',
            fontsize=8,
            titlefontsize=12):
        self.object = object
        self.filename = filename
        self.path = path
        self.x = getattr(object, "para")
        self.xlabel = xlabel
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        os.makedirs(self.path, exist_ok=True)
    def Draw_Split_Picture(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.x, self.object.KNN, color='red', linestyle='-', label='KNN')
        ax1.scatter(self.x, self.object.KNN, color='red')
        ax1.plot(self.x, self.object.NMI, color='orange', linestyle='-', label='NMI')
        ax1.scatter(self.x, self.object.NMI, color='orange')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='y')
        ax1.set_xlabel(self.xlabel, fontsize=self.titlefontsize)
        ax2.plot(self.x, self.object.SHS, color='blue', linestyle='-', label='SHS')
        ax2.scatter(self.x, self.object.SHS, color='blue')
        ax2.set_ylim(-1, 1)
        ax2.tick_params(axis='y')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=self.fontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
# 折线图绘制
class CustomLegendHandler:
    def __init__(self, color, marker, linestyle, label, markeredgecolor, markerfacecolor):
        self.color = color
        self.marker = marker
        self.linestyle = linestyle
        self.label = label
        self.markeredgecolor = markeredgecolor
        self.markerfacecolor = markerfacecolor

    def create_proxy_artist(self, legend):
        return Line2D([0], [0], color=self.color, marker=self.marker, linestyle=self.linestyle, markeredgecolor=self.markeredgecolor, markerfacecolor=self.markerfacecolor)
class Draw_Line_Chart():
    def __init__(self, filename='default', path='./Figure/',
                 fontsize=8, titlefontsize=12,
                 left=None, right=None, column=None,
                 left_label=None,
                 right_label=None,
                 ylim_left=(0,1),
                 ylim_right=(0,1),
                 left_color=None,
                 right_color=None,
                 left_marker=None,
                 right_marker=None,
                 left_markeredgecolor=None, left_markerfacecolor=None,
                 right_markeredgecolor=None, right_markerfacecolor=None,
                 xlabel='% size of train set',
                 ylabel_left='classification accuracy',
                 ylabel_right='clustering score', title = None):
        self.filename = filename
        self.path = path
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.left = np.array(left)
        self.right = np.array(right)
        self.column = np.array(column)
        self.left_num = 0 if left is None else len(left)
        self.right_num = 0 if right is None else len(right)
        self.left_label = ["left-"+str(i+1) for i in range(self.left_num)] if left_label is None else left_label
        self.right_label = ["right-"+str(i+1) for i in range(self.right_num)] if right_label is None else right_label
        self.left_color = np.random.random((self.left_num, 3)) if left_color is None else left_color
        self.right_color = np.random.random((self.right_num, 3)) if right_color is None else right_color
        self.left_markeredgecolor = self.left_color if left_markeredgecolor is None else left_markeredgecolor
        self.left_markerfacecolor = self.left_color if left_markerfacecolor is None else left_markerfacecolor
        self.right_markeredgecolor = self.right_color if right_markeredgecolor is None else right_markeredgecolor
        self.right_markerfacecolor = self.right_color if right_markerfacecolor is None else right_markerfacecolor
        self.left_marker = ["o"]*self.left_num if left_marker is None else left_marker
        self.right_marker = ["o"]*self.right_num if right_marker is None else right_marker
        self.xlabel = xlabel
        self.ylabel_left = ylabel_left
        self.ylabel_right = ylabel_right
        self.ylim_left = ylim_left
        self.ylim_right = ylim_right
        self.title = title
        os.makedirs(self.path, exist_ok=True)

    def Draw_double_line(self):
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        self.column = np.array(range(1, self.left.shape[1] + 1)) if self.column is None else self.column
        for i in range(self.left_num):
            ax1.plot(self.column, self.left[i], color=self.left_color[i],linestyle='-', label=self.left_label[i])
            ax1.scatter(self.column, self.left[i], facecolor=self.left_markerfacecolor[i], edgecolor=self.left_markeredgecolor[i], marker=self.left_marker[i])
        if self.ylim_left is not None:
            ax1.set_ylim(self.ylim_left[0], self.ylim_left[1])
        ax1.tick_params(axis='y')
        ax1.set_xlabel(self.xlabel, fontsize = self.titlefontsize)
        ax1.set_ylabel(self.ylabel_left, fontsize = self.titlefontsize)
        for i in range(self.right_num):
            ax2.plot(self.column, self.right[i], color=self.right_color[i],linestyle='--', label=self.right_label[i])
            ax2.scatter(self.column, self.right[i], facecolor=self.right_markerfacecolor[i], edgecolor=self.right_markeredgecolor[i], marker=self.right_marker[i])
        if self.ylim_right is not None:
            ax2.set_ylim(self.ylim_right[0], self.ylim_right[1])
        ax2.tick_params(axis='y')
        ax2.set_ylabel(self.ylabel_right, fontsize = self.titlefontsize)
        custom_legend_handles = []
        for color, marker, linestyle, label, markeredgecolor, markerfacecolor in zip(self.left_color, self.left_marker, ['-'] * self.left_num, self.left_label, self.left_markeredgecolor, self.left_markerfacecolor):
            custom_legend_handles.append(CustomLegendHandler(color, marker, linestyle, label, markeredgecolor, markerfacecolor))
        for color, marker, linestyle, label, markeredgecolor, markerfacecolor in zip(self.right_color, self.right_marker, ['--'] * self.right_num, self.right_label, self.right_markeredgecolor, self.right_markerfacecolor):
            custom_legend_handles.append(CustomLegendHandler(color, marker, linestyle, label, markeredgecolor, markerfacecolor))
        proxy_artists = [handler.create_proxy_artist(ax2) for handler in custom_legend_handles]
        ax2.legend(proxy_artists, self.left_label + self.right_label, loc='lower right', fontsize = self.fontsize)
        plt.xticks(fontsize = self.fontsize)
        plt.yticks(fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
    def Draw_simple_line(self):
        self.column = np.array(range(1, self.left.shape[1] + 1)) if self.column is None else self.column
        for i in range(self.left_num):
            plt.plot(self.column, self.left[i], color=self.left_color[i], linestyle='-', label=self.left_label[i])
            plt.scatter(self.column, self.left[i], facecolor=self.left_markerfacecolor[i], edgecolor=self.left_markeredgecolor[i], marker=self.left_marker[i])
        if self.ylim_left is not None:
            plt.ylim(self.ylim_left[0], self.ylim_left[1])
        plt.tick_params(axis='y')
        plt.xlabel(self.xlabel, fontsize = self.titlefontsize)
        plt.ylabel(self.ylabel_left, fontsize = self.titlefontsize)
        plt.legend(loc='lower right', fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
# 混淆矩阵绘制
class Confusion_Matrix():
    def __init__(self, filename='default', path='./Figure/',
                 fontsize=8, titlefontsize=12, title=None,
                 lgd = None):
        self.filename = filename
        self.path = path
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.lgd = lgd
        os.makedirs(self.path, exist_ok=True)

    def Drawing(self, true_label, predict_label):
        self.true_label = true_label
        self.predict_label = predict_label
        self.Calculate()
        cmap = ListedColormap(['#3564A7', '#FFFFFF', '#759FCC'])
        unique_categories, category_counts = np.unique(true_label, return_counts=True)
        if self.lgd is not None:
            xticklabels = self.lgd
            yticklabels = self.lgd
        else:
            xticklabels = yticklabels = [f"Class {i}" for i in unique_categories]
        plt.figure()
        sns.heatmap(self.color_matrix, annot=False, fmt='d',
            cbar=False, cmap=cmap,
            linecolor='#7F7F7F', linewidths=0.5)
        plt.xticks(ticks=np.array(range(len(xticklabels)))+0.5, labels=xticklabels, fontsize = self.fontsize)
        plt.yticks(ticks=np.array(range(len(yticklabels)))+0.5, labels=yticklabels, fontsize = self.fontsize, rotation=0)
        plt.xlabel("Predict Class", fontsize = self.titlefontsize)
        plt.ylabel("True Class", fontsize = self.titlefontsize)
        plt.text(self.num_classes + 0.5, self.num_classes + 0.5, f"{self.total_ratio * 100:.2f}%", fontsize=self.fontsize, horizontalalignment='center', verticalalignment='center', color='white')
        for i in range(len(self.cm)):
            for j in range(len(self.cm)):
                text = f"{self.cm[i, j]}\n({self.cell_proportion[i, j] * 100:.2f}%)"
                plt.text(j + 0.5, i + 0.5, text, fontsize=self.fontsize, horizontalalignment='center', verticalalignment='center', color='white' if i == j else 'black')
        for i in range(self.num_classes):
            plt.text(self.num_classes + 0.5, i + 0.5, f"{self.row_ratios[i] * 100:.2f}%", fontsize=self.fontsize, horizontalalignment='center', verticalalignment='center', color='white')
            plt.text(i + 0.5, self.num_classes + 0.5, f"{self.column_ratios[i] * 100:.2f}%", fontsize=self.fontsize, horizontalalignment='center', verticalalignment='center', color='white')
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
    def Calculate(self):
        self.cm = confusion_matrix(self.true_label, self.predict_label)
        self.num_classes = len(self.cm)
        self.cell_proportion = self.cm / self.cm.sum()
        self.cm_with_ratios = np.zeros((self.num_classes + 1, self.num_classes + 1))
        self.cm_with_ratios[:self.num_classes, :self.num_classes] = self.cm
        self.column_ratios = np.diag(self.cm_with_ratios)[:-1] / self.cm.sum(axis=0)
        self.row_ratios = np.diag(self.cm_with_ratios)[:-1] / self.cm.sum(axis=1)
        self.total_ratio = np.sum(np.diag(self.cm_with_ratios)) / self.cm.sum()
        self.color_matrix = np.eye(len(self.cm_with_ratios))
        self.color_matrix[-1, :] = -1
        self.color_matrix[:, -1] = -1
        self.color_matrix += 1

################################################################################
# 热图绘制
class Annotated_Heatmaps:
    def __init__(self, filename='default', path='./Figure/',
                 fontsize=8, titlefontsize=12, title=None,
                 xlabel = None, ylabel = None, cmap=('#FFFFFF', '#3564A7'),
                 xticklabels=None, yticklabels=None):
        self.filename = filename
        self.path = path
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xticklabels = xticklabels
        self.yticklabels = yticklabels
        self.cmap = cmap
        os.makedirs(self.path, exist_ok=True)

    def Drawing(self, harvest):
        if isinstance(self.cmap, tuple) and len(self.cmap) == 2:
            cmap = create_colormap(bottom=self.cmap[0], top=self.cmap[1])
        else:
            cmap = self.cmap
        if self.xticklabels is None:
            self.xticklabels = [str(i) for i in range(harvest.shape[1])]
            self.yticklabels = [str(i) for i in range(harvest.shape[0])]
        plt.figure()
        sns.heatmap(harvest, annot=False, fmt='d',
            cbar=False, cmap=cmap,
            linecolor='#7F7F7F', linewidths=0.5)
        plt.xticks(ticks=np.array(range(len(self.xticklabels)))+0.5, labels=self.xticklabels, fontsize = self.fontsize)
        plt.yticks(ticks=np.array(range(len(self.yticklabels)))+0.5, labels=self.yticklabels, fontsize = self.fontsize, rotation=0)
        if self.xlabel is not None:
            plt.xlabel(self.xlabel, fontsize = self.titlefontsize)
        if self.ylabel is not None:
            plt.ylabel(self.ylabel, fontsize = self.titlefontsize)
        for i in range(harvest.shape[0]):
            for j in range(harvest.shape[1]):
                text = f"({harvest[i, j] * 100:.2f}%)"
                plt.text(j + 0.5, i + 0.5, text, fontsize=self.fontsize,
                         horizontalalignment='center', verticalalignment='center',
                         color='black')
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
# 颜色图绘制
class Color_Mapping():
    def __init__(self, filename='default', path='./Figure/',
                 fontsize=8, titlefontsize=12, title = None,
                 lgd=None):
        self.filename = filename
        self.path = path
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.lgd = lgd
        os.makedirs(self.path, exist_ok=True)
    def Calculate(self):
        self.unique_categories, self.category_counts = np.unique(self.true_label, return_counts=True)
        self.matrix = np.zeros((len(self.unique_categories), self.category_counts[0]))
        for c, category in enumerate(self.unique_categories):
            indices = np.where(self.true_label == category)[0]
            label = self.predict_label[indices]
            self.matrix[c, :] = label
    def Mapping(self, true_label, predict_label):
        self.true_label = true_label
        self.predict_label = predict_label
        self.Calculate()
        plt.figure()
        plt.imshow(self.matrix, cmap='viridis', aspect='auto', interpolation='nearest')
        if self.lgd is not None:
            yticklabels = self.lgd
        else:
            yticklabels = [f"Class {i}" for i in self.unique_categories]
        plt.ylabel("True Class", fontsize = self.titlefontsize)
        plt.yticks(ticks=list(range(len(yticklabels))), labels=yticklabels, fontsize = self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize = self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
################################################################################
# 可视化数据集
class Visual_Pixes():
    def __init__(
            self, pic_height=32, pic_weight=32, total_weight=10,
            fontsize=8, titlefontsize=12,
            path="./Figure/", filename="Visualization-Datssets", lgd = None,
            title= None):
        self.pic_height = pic_height
        self.pic_weight = pic_weight
        self.total_weight = total_weight
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.path = path
        self.filename = filename
        self.lgd = lgd
        self.title= title
        os.makedirs(path, exist_ok=True)
    def Calculate_Pixes(self, data, target):
        data = data.reshape(data.shape[0], self.pic_height, self.pic_weight)
        self.unique_categories, self.category_counts = np.unique(target, return_counts=True)
        self.n = len(self.unique_categories)
        vline = np.ones((self.pic_height, 1))
        temph = np.ones((1, self.total_weight*(self.pic_weight+1)+1))
        hline = np.ones((1, self.total_weight*(self.pic_weight+1)+1))
        m = len(data)//self.n
        for i in range(self.n):
            tempv = np.zeros((self.pic_height, 1))
            for j in range(self.total_weight):
                tempv = np.concatenate((tempv, data[j + m * i]), axis=1)
                tempv = np.concatenate((tempv, vline), axis=1)
            temph = np.concatenate((temph, tempv), axis=0)
            temph = np.concatenate((temph, hline), axis=0)
        return temph
    def Drawing(self, data, target):
        matrix = self.Calculate_Pixes(data, target)
        plt.imshow(matrix)
        if self.lgd is not None:
            yticklabels = self.lgd
        else:
            yticklabels = [f"Class {i}" for i in self.unique_categories]
        xticklabels = [f"Sample {i}" for i in range(1, self.total_weight+1)]
        plt.xticks(ticks=list(range(1+self.pic_weight//2, (self.pic_weight+1)*self.total_weight+1, self.pic_weight+1)), labels=xticklabels, fontsize=self.fontsize, rotation=30)
        plt.yticks(ticks=list(range(1+self.pic_height//2, (self.pic_height+1)*self.n+1, self.pic_height+1)), labels=yticklabels, fontsize=self.fontsize)
        if self.title is not None:
            plt.title(self.title, fontsize=self.titlefontsize)
        Save_Pictures(os.path.join(self.path, self.filename))

################################################################################
# 创建自定义colormap
def create_colormap(bottom="blue", top="yellow"):
    colors = [bottom, top]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
    return cmap

################################################################################
# 3D直方图
class Draw_Bars:
    def __init__(self):
        pass
    def Draw_Bars12(self, matrix, xticklabels=None, yticklabels=None, zticklabels=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=20, azim=-45)
        xtick = np.arange(matrix.shape[1])
        ytick = np.arange(matrix.shape[0])
        ztick = np.arange(0, 12, 2) * 0.10
        if xticklabels is None:
            xticklabels = [str(i) for i in xtick]
        if yticklabels is None:
            yticklabels = [str(i) for i in ytick]
        if zticklabels is None:
            zticklabels = ["{:.1f}".format(i) for i in ztick]
        xtick_mesh, ytick_mesh = np.meshgrid(xtick, ytick)
        top = matrix
        bottom = np.zeros_like(top)
        width = 0.8
        colors = plt.cm.viridis(np.linspace(0, 1, matrix.shape[0]))
        for i in range(0, matrix.shape[0]):
            ax.bar3d(xtick_mesh[i], ytick_mesh[i], bottom[i],
                     width, width, top[i], color=colors[i],
                     edgecolor='black', linewidth=0.2,
                     shade=False, alpha=1)
        ax.set_title('Shaded', fontsize=12)
        ax.grid(False)
        ax.set_xlabel("x label", fontsize=12, labelpad=-6)
        ax.set_ylabel("y label", fontsize=12, labelpad=-6)
        xtick =  [xt + 0.5 for xt in xtick]
        ytick =  [yt + 0.5 for yt in ytick]
        ax.set_xticks(xtick, xticklabels)
        ax.set_yticks(ytick, yticklabels)
        ax.set_zticks(ztick, zticklabels)
        ax.tick_params(axis='x', pad=-5, labelsize=8)
        ax.tick_params(axis='y', pad=-5, labelsize=8)
        ax.tick_params(axis='z', pad=-1, labelsize=8)
        Save_Pictures()

    def Draw_Bars_3D(self, matrix, xticklabels=None, yticklabels=None, zticklabels=None):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        xtick = np.arange(matrix.shape[1])
        ytick = np.arange(matrix.shape[0])
        ztick = np.arange(0, 12, 2) * 0.10
        if xticklabels is None:
            xticklabels = [str(i) for i in xtick]
        if yticklabels is None:
            yticklabels = [str(i) for i in ytick]
        if zticklabels is None:
            zticklabels = ["{:.1f}".format(i) for i in ztick]
        xtick_mesh, ytick_mesh = np.meshgrid(xtick, ytick)
        x, y = xtick_mesh.ravel(), ytick_mesh.ravel()
        top = matrix.ravel()
        bottom = np.zeros_like(x)
        width = 0.8
        custom_cmap = create_colormap()
        colors = custom_cmap(top / top.max())
        ax.bar3d(x, y, bottom, width, width, top, color=colors, cmap=custom_cmap)
        ax.set_title('Shaded', fontsize=12)
        ax.grid(False)
        ax.set_xlabel("x label", fontsize=12, labelpad=-6)
        ax.set_ylabel("y label", fontsize=12, labelpad=-6)
        xtick = [xt + 0.5 for xt in xtick]
        ytick = [yt + 0.5 for yt in ytick]
        ax.set_xticks(xtick, xticklabels)
        ax.set_yticks(ytick, yticklabels)
        ax.set_zticks(ztick, zticklabels)
        ax.tick_params(axis='x', pad=-5, labelsize=8)
        ax.tick_params(axis='y', pad=-5, labelsize=8)
        ax.tick_params(axis='z', pad=-1, labelsize=8)
        Save_Pictures()

################################################################################
# 误差图绘制
class Error_Drawing:
    def __init__(
            self, filename='default', path='./Figure/', fontsize=8,
            titlefontsize=12, title=None, xlabel="x", ylabel="y"
    ):
        self.filename = filename
        self.path = path
        self.fontsize = fontsize
        self.titlefontsize = titlefontsize
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        os.makedirs(self.path, exist_ok=True)

    def drawing_banding(self, x_value, mean_value, std_value, colors=None, markers=None, labels=None):
        mean_value = np.array(mean_value)
        std_value = np.array(std_value)
        if len(mean_value.shape) == 1 and len(mean_value.shape):
            mean_value = mean_value.reshape((1, -1))
            std_value = std_value.reshape((1, -1))
        colors = np.random.random((mean_value.shape[1], 3)) if colors is None else colors
        markers = ["o"] * mean_value.shape[1] if markers is None else markers
        labels = ["model"] * mean_value.shape[1] if labels is None else labels
        plt.figure()
        for i in range(mean_value.shape[0]):
            plt.plot(
                x_value, mean_value[i], label="mean of " + labels[i],
                color=colors[i], marker=markers[i])
            plt.fill_between(
                x_value, mean_value[i] - std_value[i],
                mean_value[i] + std_value[i], color=colors[i],
                alpha=0.2, label="std of " + labels[i])
        plt.xlabel(self.xlabel, fontsize=self.titlefontsize)
        plt.ylabel(self.ylabel, fontsize=self.titlefontsize)
        plt.ylim(0, 1.05)
        plt.title(self.title, fontsize=self.titlefontsize)
        plt.legend(fontsize=self.fontsize)
        plt.grid(True)
        Save_Pictures(os.path.join(self.path, self.filename))

    def drawing_line_error(self, x_value, mean_value, std_value, colors=None, labels=None):
        mean_value = np.array(mean_value)
        std_value = np.array(std_value)
        if len(mean_value.shape) == 1 and len(mean_value.shape):
            mean_value = mean_value.reshape((1, -1))
            std_value = std_value.reshape((1, -1))
        colors = np.random.random((mean_value.shape[1], 3)) if colors is None else colors
        labels = ["model"] * mean_value.shape[1] if labels is None else labels
        plt.figure()
        for i in range(mean_value.shape[0]):
            plt.errorbar(
                x_value, mean_value[i], yerr=std_value[i],
                color=colors[i], label=labels[i])
        plt.xlabel(self.xlabel, fontsize=self.titlefontsize)
        plt.ylabel(self.ylabel, fontsize=self.titlefontsize)
        plt.ylim(0, 1.05)
        plt.title(self.title, fontsize=self.titlefontsize)
        plt.legend(fontsize=self.fontsize)
        plt.grid(True)
        Save_Pictures(os.path.join(self.path, self.filename))

    def drawing_bar_error(self, x_value, mean_value, std_value, colors=None, labels=None):
        mean_value = np.array(mean_value)
        std_value = np.array(std_value)
        if len(mean_value.shape) == 1 and len(mean_value.shape):
            mean_value = mean_value.reshape((1, -1))
            std_value = std_value.reshape((1, -1))
        colors = np.random.random((mean_value.shape[1], 3)) if colors is None else colors
        labels = ["model"] * mean_value.shape[1] if labels is None else labels
        x = np.arange(mean_value.shape[1])
        width = 0.80 / mean_value.shape[0]
        fig, ax = plt.subplots(layout='constrained')
        for i in range(mean_value.shape[0]):
            ax.bar(x + width * i, mean_value[i], width, yerr=std_value[i], color=colors[i], label=labels[i])
        ax.set_xlabel(self.xlabel, fontsize=self.titlefontsize)
        ax.set_ylabel(self.ylabel, fontsize=self.titlefontsize)
        ax.set_ylim(0, 1.05)
        ax.set_title(self.title, fontsize=self.titlefontsize)
        ax.set_xticks(x+0.4-width/2, x_value, fontsize=self.fontsize)
        plt.legend(fontsize=self.fontsize)
        Save_Pictures(os.path.join(self.path, self.filename))

    def drawing_barh_error(self, x_value, mean_value, std_value, colors=None, labels=None):
        mean_value = np.array(mean_value)
        std_value = np.array(std_value)
        if len(mean_value.shape) == 1 and len(mean_value.shape):
            mean_value = mean_value.reshape((1, -1))
            std_value = std_value.reshape((1, -1))
        colors = np.random.random((mean_value.shape[1], 3)) if colors is None else colors
        labels = ["model"] * mean_value.shape[1] if labels is None else labels
        x = np.arange(mean_value.shape[1])
        width = 0.80 / mean_value.shape[0]
        fig, ax = plt.subplots(layout='constrained')
        for i in range(mean_value.shape[0]):
            ax.barh(x + width * i, mean_value[i], width, xerr=std_value[i], color=colors[i], label=labels[i])
        ax.set_xlabel(self.ylabel, fontsize=self.titlefontsize)
        ax.set_ylabel(self.xlabel, fontsize=self.titlefontsize)
        ax.set_title(self.title, fontsize=self.titlefontsize)
        ax.set_yticks(x+0.4-width/2, x_value, fontsize=self.fontsize)
        ax.set_xlim(0, 1.05)
        plt.legend(fontsize=self.fontsize)
        Save_Pictures(os.path.join(self.path, self.filename))
