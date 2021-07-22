import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

'''良性/恶性乳腺癌肿瘤数据预处理'''
# 创建特征列表
column_names = ['Sample code number', 1, 2, 3, 4, 5, 6, 7, 8, 9, 'class']
# 读取原始乳腺癌肿瘤数据
data = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)
# 将?替换为标准缺失值np.nan表示
data=data.replace(to_replace='?', value=np.nan)
# 删除所有含缺失值的行
data = data.dropna(how='any')
print(data.shape)
# 无缺失值数据683条，每条样本包含检索ID一个+9个医学特征+一个肿瘤分类
print(data.head())

'''分割良性/恶性乳腺癌肿瘤训练、测试数据 3:1'''
x_train, x_test, y_train, y_test = train_test_split(
    data[column_names[1:10]],
    data[column_names[10]],
    test_size=0.25,
    random_state=11)
print('训练样本良性2/恶性4分布：', '\n', y_train.value_counts(), '\n',
      '测试样本良性2/恶性4分布：', '\n', y_test.value_counts())

'''使用线性分类模型（Logistic回归与随机梯度参数估计）预测良性/恶性肿瘤'''
# 标准化数据，保证每个维度的特征数据方差为1，均值为0，使得预测结果不会被某些维度过大的特征值主导
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
# 初始化LogisticRegression、SGDClassifier
lr = LogisticRegression()
SgdC = SGDClassifier()
# 调用LogisticRegression中的fit函数/模块用来训练模型参数
lr.fit(x_train, y_train)
# 使用训练好的模型lr对测试集进行预测
lr_y_predict = lr.predict(x_test)
# 调用SGDClassifier中的fit函数/模块用来训练模型参数
SgdC.fit(x_train, y_train)
# 使用训练好的模型lr对测试集进行预测
SgdC_y_predict = SgdC.predict(x_test)

'''使用线性分类器模型分析之前的预测性能'''
# 分别使用Logistic模型和随机梯度下降模型自带的评分函数score获得模型在测试集上的准确性结果
print('Accuracy of LR Classifier:', '\n', lr.score(x_test, y_test), '\n',
      'Accuracy of SGD Classifier:', '\n', SgdC.score(x_test, y_test))
# 分别使用Logistic模型和随机梯度下降模型获得精确率&召回率&F1指标三个指标结果
print('precision&recall&f1-score of LR Classifier:', '\n',
      classification_report(y_test, lr_y_predict, target_names=['良性', '恶性']), '\n',
      'precision&recall&f1-score of SGD Classifier:', '\n',
      classification_report(y_test, SgdC_y_predict, target_names=['良性', '恶性'])
      )

# 求取两种模型预测结果的混淆矩阵
cmLrDf = pd.DataFrame(confusion_matrix(y_test, lr_y_predict),
                      index=['良性', '恶性'],
                      columns=['良性', '恶性'])
cmSgdCDf = pd.DataFrame(confusion_matrix(y_test, SgdC_y_predict),
                        cmLrDf.index, cmLrDf.columns)
print('LR混淆矩阵：', '\n', cmLrDf, '\n',
      'SgdC混淆矩阵：', '\n', cmSgdCDf)
# 设置Seaborn绘图风格默认
sns.set()
plt.figure(1)
sns.heatmap(cmLrDf, annot=True, fmt="d",
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('predict'), plt.ylabel('true'), plt.title('Logistic Model')
plt.figure(2)
sns.heatmap(cmSgdCDf, annot=True, fmt="d",
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('predict'), plt.ylabel('true'), plt.title('SGD Model')
plt.show()










