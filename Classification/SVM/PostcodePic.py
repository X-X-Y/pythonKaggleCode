import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 加载手写体数字的数码图像数据
digits = load_digits()
print(digits.data.shape)
print(digits.data[0, :])
# 分割训练集和测试集 3:1
x_train, x_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.25, random_state=22)
# 数据标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
# 初始化线性假设的支持向量机分类器
linSvc = LinearSVC()
# 进行模型训练
linSvc.fit(x_train, y_train)
# 利用训练好的模型对测试样本的数字类别进行预估
y_predict = linSvc.predict(x_test)
# 准确性测评
print('The Accuracy of Linear SVC is', linSvc.score(x_test, y_test))
# 对预测结果做更详细的分析
print('precision&recall&f1-score of Linear SVC:', '\n',
      classification_report(y_test, y_predict,
                            target_names=digits.target_names.astype(str)))

# 求取SVM模型预测结果的混淆矩阵
cmSVC = pd.DataFrame(confusion_matrix(y_test, y_predict),
                     index=digits.target_names.astype(str),
                     columns=digits.target_names.astype(str))
print('SVM混淆矩阵：', '\n', cmSVC)
# 设置Seaborn绘图风格默认
sns.set()
plt.figure(1)
sns.heatmap(cmSVC, annot=True, fmt="d",
            xticklabels=digits.target_names.astype(str),
            yticklabels=digits.target_names.astype(str))
plt.xlabel('predict'), plt.ylabel('true'), plt.title('LinearSVC Model')
plt.show()





