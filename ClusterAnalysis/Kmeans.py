import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

'''K-means算法在手写体数字图像数据上的使用示例'''
'''
# 读取训练集和测试集
digits_train = pd.read_csv('optdigits.tra', header=None)
digits_test = pd.read_csv('optdigits.tes', header=None)
# 分离出64维度像素特征和1维度数字目标
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]
x_test = digits_test[np.arange(64)]
y_test = digits_test[64]
# 初始化KMeans模型，并设置聚类中心数量为10
kmeans = KMeans(n_clusters=10)
kmeans.fit(x_train)
# 逐条判断每个测试图像所属的聚类中心
y_pred = kmeans.predict(x_test)
# 因为该图片数据自身带有正确的类别信息，使用ARI进行K-means聚类性能评估
print('The ARI of KMeans is', metrics.adjusted_rand_score(y_test, y_pred))
'''


'''利用轮廓系数评价不同类簇数量的K-means聚类实例'''
from sklearn.metrics import silhouette_score

plt.figure(1)
plt.subplot(3, 2, 1)
x1 = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
x2 = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
x = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
# 在一号子图做出原始信号点阵分布
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Instances')
plt.scatter(x1, x2)
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
clusters = [2, 3, 4, 5, 8]
subplot_counters = 1
sc_scores = []
for t in clusters:
    subplot_counters += 1
    plt.subplot(3, 2, subplot_counters)
    kmeans_model = KMeans(n_clusters=t).fit(x)
    print(kmeans_model.labels_)
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    sc_score = silhouette_score(x, kmeans_model.labels_,metric='euclidean')
    sc_scores.append(sc_score)
    plt.title('K=%s, silhouette coefficient=%0.03f' % (t, sc_score))


'''“肘部”观察法示例'''
from scipy.spatial.distance import cdist
cluster1 = np.random.uniform(0.5, 1.5, (2, 10))
cluster2 = np.random.uniform(5.5, 6.5, (2, 10))
cluster3 = np.random.uniform(3.0, 4.0, (2, 10))
# 绘制30个数据样本的分布图像，30行2列
x = np.hstack((cluster1, cluster2, cluster3)).T
plt.figure(2)
plt.subplot(1, 2, 1)
plt.title('Instances')
plt.scatter(x[:, 0], x[:, 1])
plt.xlabel('x1')
plt.ylabel('x2')
# 测试9种不同聚类中心数量下，每组情况的聚类质量
K = range(1, 10)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    meandistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_,
                                            'euclidean'), axis=1))/x.shape[0])
plt.subplot(1, 2, 2)
plt.plot(K, meandistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average Dispersion')
plt.title('Selecting k with the Elbow Method')
plt.show()











