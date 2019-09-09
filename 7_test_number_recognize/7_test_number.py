# 头文件
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Activation, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import Adam


# 读取数据并查看，数据集下载地址https://d.ailemon.me/mnist.npz
# 也可以使用 (x_train, y_train), (x_test, y_test) = mnist.load_data()会自动在线下载


(x_train, y_train), (x_test, y_test) = mnist.load_data("../mnist.npz")

# print("x_train",x_train.shape)
# print("y_train",y_train.shape)
# print("x_test",x_test.shape)
# print("y_test",y_test.shape)
# x_train (60000, 28, 28)
# y_train (60000,)
# x_test (10000, 28, 28)
# y_test (10000,)

plt.subplot(121)
plt.imshow(x_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(122)
plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()


# 数据预处理
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# 标签onehot编码
# to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示
y_test = np_utils.to_categorical(y_test, 10)
y_train = np_utils.to_categorical(y_train, 10)



# design model
model = Sequential()
model.add(Convolution2D(25, (5, 5), input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))
model.add(Convolution2D(50, (5, 5)))
model.add(MaxPooling2D(2, 2))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))
adam = Adam(lr=0.001)

# compile model
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# training model
model.fit(x_train, y_train, batch_size=100, epochs=5)

# test model
print(model.evaluate(x_test, y_test, batch_size=100))j

# save model
model.save('./model.h5')





