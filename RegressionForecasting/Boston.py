import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

boston = load_boston()
x = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.25,
                                                    random_state=33)
print('The max target value is', np.max(boston.target), '\n',
      'The min target value is', np.min(boston.target), '\n',
      'The average target value is', np.mean(boston.target))
# 预测目标房价之间的差异较大，对特征和目标值进行标准化处理(可还原真实结果)
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

'''线性回归器'''
# 使用线性回归模型LinearRegression和SGDRegressor分别对波士顿房价数据进行训练学习及预测
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
sgdr = SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)
# 评价LinearRegression和SGDRegressor两种模型性能
# LinearRegression模型自带评估模块、R2、MSE均方误差和MAE平均绝对误差的评估结果
print('The value of default measurement of LinearRegression is',
      lr.score(x_test, y_test), '\n',
      'The value of R-squared of LinearRegression is',
      r2_score(y_test, lr_y_predict), '\n',
      'The mean squared error of LinearRegression is',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(lr_y_predict)), '\n',
      'The mean absoluate error of LinearRegression is',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(lr_y_predict))
      )
# SGDRegressor模型自带评估模块、R2、MSE均方误差和MAE平均绝对误差评估结果
print('The value of default measurement of SGDRegressor is',
      sgdr.score(x_test, y_test), '\n',
      'The value of R-squared of SGDRegressor is',
      r2_score(y_test, sgdr_y_predict), '\n',
      'The mean squared error of SGDRegressor is',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(sgdr_y_predict)), '\n',
      'The mean absoluate error of SGDRegressor is',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(sgdr_y_predict))
      )

'''支持向量机(回归)
from sklearn.svm import SVR
# 使用线性核函数配置的支持向量机进行回归训练和预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)
# 使用多项式核函数配置的支持向量机进行回归训练和预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train)
poly_svr_y_predict = poly_svr.predict(x_test)
# 使用径向基核函数配置的支持向量机进行回归训练和预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)
# 使用R2、MSE均方误差和MAE平均绝对误差对三种配置支持向量机(回归)模型在相同测试集上进行性能评估
print('The value of R-squared of Linear SVR is',
      linear_svr.score(x_test, y_test), '\n',
      'The mean squared error of Linear SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(linear_svr_y_predict)), '\n',
      'The mean absoluate error of Linear SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(linear_svr_y_predict))
      )
print('The value of R-squared of poly SVR is',
      poly_svr.score(x_test, y_test), '\n',
      'The mean squared error of poly SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(poly_svr_y_predict)), '\n',
      'The mean absoluate error of poly SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(poly_svr_y_predict))
      )
print('The value of R-squared of rbf SVR is',
      rbf_svr.score(x_test, y_test), '\n',
      'The mean squared error of rbf SVR is',
      mean_squared_error(ss_y.inverse_transform(y_test),
                         ss_y.inverse_transform(rbf_svr_y_predict)), '\n',
      'The mean absoluate error of rbf SVR is',
      mean_absolute_error(ss_y.inverse_transform(y_test),
                          ss_y.inverse_transform(rbf_svr_y_predict))
      )
'''








