import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 输入样本特征以及目标值，即披萨直径
x_train = [[6], [8], [10], [14], [18]]
y_train = [[7], [9], [13], [17.5], [18]]
# 准备测试数据
x_test = [[6], [8], [11], [16]]
y_test = [[8], [12], [15], [18]]
poly4 = PolynomialFeatures(degree=4)
# 映射出4次多项式特征
x_train_poly4 = poly4.fit_transform(x_train)
regressor_poly4 = LinearRegression()
regressor_poly4.fit(x_train_poly4, y_train)
# 使用测试数据对4次多项式回归模型的性能进行评估
x_test_poly4 = poly4.transform(x_test)
print(regressor_poly4.score(x_test_poly4, y_test), '\n',
      '普通4次多项式回归模型的参数列表:', regressor_poly4.coef_)

'''L1范数正则化，Lasso模型在4次多项式特征上的拟合表现'''
from sklearn.linear_model import Lasso
# 使用Lasso对4次多项式进行拟合
lasso_poly4 = Lasso()
lasso_poly4.fit(x_train_poly4, y_train)
print(lasso_poly4.score(x_test_poly4, y_test), '\n',
      'Lasso模型的参数列表:', lasso_poly4.coef_)

'''L2范数正则化，Ridge模型在4次多项式特征上的拟合表现'''
from sklearn.linear_model import Ridge
# 使用Lasso对4次多项式进行拟合
ridge_poly4 = Ridge()
ridge_poly4.fit(x_train_poly4, y_train)
print(ridge_poly4.score(x_test_poly4, y_test), '\n',
      'Lasso模型的参数列表:', ridge_poly4.coef_, '\n',
      'Ridge模型拟合后参数的平方和：', np.sum(ridge_poly4.coef_**2))






