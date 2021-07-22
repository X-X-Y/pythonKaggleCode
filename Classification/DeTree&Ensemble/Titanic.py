import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
titanic = pd.read_csv('titanic.txt')
print(titanic.head())
print(titanic.info())

# 人工选取pclass、age以及sex作为判别乘客是否能够生还的特征
x = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 数据预处理，对age特征缺失的数据使用平均值代替
x['age'].fillna(x['age'].mean(), inplace=True)
# 分割训练集和测试集 3:1
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=25)
# 对类别型特征进行转化，成为特征向量
vec = DictVectorizer(sparse=False)
x_train = vec.fit_transform(x_train.to_dict(orient='records'))
x_test = vec.fit_transform(x_test.to_dict(orient='records'))
print(vec.feature_names_)
# 使用单一决策树进行训练及预测
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_pred = dtc.predict(x_test)
# 使用随机森林分类器训练及预测
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)
# 使用梯度提升决策树进行训练及预测
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_y_pred = gbc.predict(x_test)
# 三种模型对泰坦尼克号数据预测性能
# 准确性测评
print('The Accuracy of DecisionTree&RandomForestClassifier&GradientTreeBoosting is', '\n',
      dtc.score(x_test, y_test), '\n',
      rfc.score(x_test, y_test), '\n',
      gbc.score(x_test, y_test))
# 对预测结果做更详细的分析
print('precision&recall&f1-score of '
      'DecisionTree&RandomForestClassifier&GradientTreeBoosting:', '\n',
      classification_report(y_test, dtc_y_pred, target_names=['died', 'survived']), '\n',
      classification_report(y_test, rfc_y_pred, target_names=['died', 'survived']), '\n',
      classification_report(y_test, gbc_y_pred, target_names=['died', 'survived']))

# 求取三个模型预测结果的混淆矩阵
cmDtc = pd.DataFrame(confusion_matrix(y_test, dtc_y_pred),
                     index=['died', 'survived'],
                     columns=['died', 'survived'])
cmRfc = pd.DataFrame(confusion_matrix(y_test, rfc_y_pred),
                     index=['died', 'survived'],
                     columns=['died', 'survived'])
cmGbc = pd.DataFrame(confusion_matrix(y_test, gbc_y_pred),
                     index=['died', 'survived'],
                     columns=['died', 'survived'])
print('DecisionTree混淆矩阵：', '\n', cmDtc, '\n',
      'RandomForestClassifier混淆矩阵：', '\n', cmRfc, '\n',
      'GradientTreeBoosting混淆矩阵：', '\n', cmGbc)
# 设置Seaborn绘图风格默认
sns.set()
plt.figure(1)
sns.heatmap(cmDtc, annot=True, fmt="d",
            xticklabels=['died', 'survived'],
            yticklabels=['died', 'survived'])
plt.xlabel('predict'), plt.ylabel('true'), plt.title('DecisionTree Model')
plt.figure(2)
sns.heatmap(cmRfc, annot=True, fmt="d",
            xticklabels=['died', 'survived'],
            yticklabels=['died', 'survived'])
plt.xlabel('predict'), plt.ylabel('true'), plt.title('RandomForestClassifier Model')
plt.figure(3)
sns.heatmap(cmGbc, annot=True, fmt="d",
            xticklabels=['died', 'survived'],
            yticklabels=['died', 'survived'])
plt.xlabel('predict'), plt.ylabel('true'), plt.title('GradientTreeBoosting Model')
plt.show()











