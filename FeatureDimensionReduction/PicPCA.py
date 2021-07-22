import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

'''K-means算法在手写体数字图像数据上的使用示例'''
# 读取训练集和测试集
digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)
# 分离出64维度像素特征和1维度数字目标
x_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]
# 初始化一个压缩高维至二维的PCA
estimator = PCA(n_components=2)
x_pca = estimator.fit_transform(x_digits)
print(x_pca.shape)


def plot_pca_scatter():
    colors = ['black', 'blue', 'purple', 'yellow', 'white',
              'red', 'lime', 'cyan', 'orange', 'gray']
    for i in range(len(colors)):
        px = x_pca[:, 0][y_digits.values == i]
        py = x_pca[:, 1][y_digits.values == i]
        plt.scatter(px, py, c=colors[i])
    plt.legend(np.arange(0, 10).astype(str))
    plt.xlabel('First Principal Componpent')
    plt.xlabel('Second Principal Componpent')
    plt.show()


plot_pca_scatter()













