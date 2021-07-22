import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''使用线性回归模型在披萨训练样本上进行拟合'''
# 输入样本特征以及目标值，即披萨直径
x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx = np.linspace(0, 26, 100)
xx = xx.reshape(xx.shape[0], 1)
# 以上述100个数据点作为基准，预测回归直线
yy = regressor.predict(xx)
plt.subplot(1, 3, 1)
plt.scatter(x_train, y_train)
plt1 = plt.plot(xx, yy, label='Degree=1')
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.title('R-squared(Degree=1): %.3f' % regressor.score(x_train, y_train))
plt.legend(handles=plt1)

'''使用2次多项式回归模型在披萨训练样本上进行拟合'''
from sklearn.preprocessing import PolynomialFeatures

poly2 = PolynomialFeatures(degree=2)
# 映射出2次多项式特征
x_train_poly2 = poly2.fit_transform(x_train)
# 尽管特征维度有提升，但是模型基础仍然是线性模型
regressor_poly2 = LinearRegression()
# 对2次多项式回归模型进行训练
regressor_poly2.fit(x_train_poly2, y_train)
# 从新映射绘图用x轴采样数据
xx_poly2 = poly2.transform(xx)
yy_poly2 = regressor_poly2.predict(xx_poly2)
plt.subplot(1, 3, 2)
plt.scatter(x_train, y_train)
plt1 = plt.plot(xx, yy, label='Degree=1')
plt2 = plt.plot(xx, yy_poly2, label='Degree=2')
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.title('R-squared(Degree=2): %.3f' % regressor_poly2.score(x_train_poly2, y_train))
plt.legend(handles=plt1+plt2)

'''使用4次多项式回归模型在披萨训练样本上进行拟合'''
poly4 = PolynomialFeatures(degree=4)
x_train_poly4 = poly4.fit_transform(x_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4, y_train)
xx_poly4 = poly4.transform(xx)
yy_poly4 = regressor_poly4.predict(xx_poly4)
plt.subplot(1, 3, 3)
plt.scatter(x_train, y_train)
plt1 = plt.plot(xx, yy, label='Degree=1')
plt2 = plt.plot(xx, yy_poly2, label='Degree=2')
plt4 = plt.plot(xx, yy_poly4, label='Degree=4')
plt.axis([0, 25, 0, 25])
plt.xlabel('Diameter of Pizza')
plt.ylabel('Price of Pizza')
plt.title('R-squared(Degree=4): %.3f' % regressor_poly4.score(x_train_poly4, y_train))
plt.legend(handles=plt1+plt2+plt4)
plt.show()















