import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 导入数据
iris = load_iris()
print(iris.data.shape)
print(iris.DESCR)
# 分割训练集和测试集 3:1
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.25, random_state=24)
# 数据标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
# 训练K邻近模型，预测测试集
knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)
# 准确性测评
print('The Accuracy of K-Nearest Neighbor Classifier is', knc.score(x_test, y_test))
# 对预测结果做更详细的分析
print('precision&recall&f1-score of K-Nearest Neighbor Classifier:', '\n',
      classification_report(y_test, y_predict,
                            target_names=iris.target_names))

# 求取K邻近模型预测结果的混淆矩阵
cmKnc = pd.DataFrame(confusion_matrix(y_test, y_predict),
                     index=iris.target_names,
                     columns=iris.target_names)
print('KNeighborsClassifier混淆矩阵：', '\n', cmKnc)
# 设置Seaborn绘图风格默认
sns.set()
plt.figure(1)
sns.heatmap(cmKnc, annot=True, fmt="d",
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('predict'), plt.ylabel('true'), plt.title('KNeighborsClassifier Model')
plt.show()









