import numpy as np
from keras.models import Sequential
# Sequential 顺序的模型
from keras.layers import Dense
# Dense 构造的全连接层
from keras.optimizers import SGD
# optimizers 优化方法

# 构建数据集
X_data = np.linspace(-1,1,1000)[:, np.newaxis]
noise = np.random.normal(0,0.05,X_data.shape)
y_data = np.square(X_data) + noise + 0.5

print("shape")
print(X_data.shape)

# 构建神经网络
# 构建model
model = Sequential()
# 向model中加入层
# 这层输出有10个节点，输入第一维输入为1，初始化的方法正态分布，默认自己查一下，激活函数
model.add(Dense(10, input_shape=(1,), init='normal', activation='relu'))
#model.add(Dense(5, activation='relu'))
# vs 分类为softmax激活
model.add(Dense(1, init='normal'))
#sgd = SGD(lr=0.1)
# 指定loss函数，优化器哪种学习方法更新梯度
model.compile(loss='mean_squared_error', optimizer="sgd")
# 训练
# 训练轮次10次，也就是迭代10次，每次进入模型的样本数，verbose打印出fit过程中的参数
model.fit(X_data, y_data, nb_epoch=10, batch_size=10, verbose=1)
# 在原数据上预测
y_predict=model.predict(X_data)
#print(y_predict)
# 打出模型的每一层的参数
model.summary()
# 可视化
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(X_data, y_data)
ax.plot(X_data,y_predict,'r-',lw=5)
plt.show()

