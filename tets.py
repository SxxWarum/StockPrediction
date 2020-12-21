import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import math
from sklearn.preprocessing import MinMaxScaler


def scheduler(epoch, lr):
    if epoch < 256:
        return lr
    else:
        return lr * tf.math.exp(-1 * epoch / 512)


f_open = open('stockdata\\test.csv')

# 获取原始数据
ori_data = pd.read_csv(f_open)
# print(ori_data)
# 转换成numpy格式
ori_data_np = np.array(ori_data)

print(ori_data_np.shape)

# scalerx是第一列的缩放参数, scalery是第二列的缩放参数
# [:, np.newaxis]功能是将np从行转换到列
scalerx = MinMaxScaler()
scalery = MinMaxScaler()
scalerx.fit(ori_data_np[:, 0][:, np.newaxis])
scalery.fit(ori_data_np[:, 1][:, np.newaxis])
# ori_data_np_normalized是ori_data_np正则化以后的数据
ori_data_np_normalized = np.concatenate(
    (scalerx.fit_transform(ori_data_np[:, 0][:, np.newaxis]),
     scalery.fit_transform(ori_data_np[:, 1][:, np.newaxis])),
    axis=1)
# ori_data_np_check是检查数据, 与ori_data对比, 测试无误
ori_data_np_check = np.concatenate(
    (scalerx.inverse_transform(ori_data_np_normalized[:, 0][:, np.newaxis]),
     scalery.inverse_transform(ori_data_np_normalized[:, 1][:, np.newaxis])),
    axis=1)
# 测试数据写进data_check.csv文件, 与test.csv文件对比
ori_normalized_data_check = pd.DataFrame(ori_data_np_check)
ori_normalized_data_check.to_csv('data_check.csv')

x, y = [], []
# 3行作为一个序列, 第3行的Y作为目标值, 重新生成序列数据
# data_x的shape: [397, 3, 1]
# data_y的shape: [397, 1]
for i in range(len(ori_data) - 3):
    # 对应版本0.2.1.1 增加此行, 数据增加一个奇偶性判断
    temp = (ori_data_np_normalized[i:i + 3, 0]).extend(
        (ori_data_np[i:i + 3, :1].sum() % 2).reshape(1, 1, 1))
    x.append(ori_data_np_normalized[i:i + 3, :1])
    y.append(ori_data_np_normalized[i + 2, 1:])
data_x = np.array(x).astype(np.float32)
data_y = np.array(y).astype(np.float32)
print(data_x.shape)
print(data_y.shape)

# 缩放数据
train_end_index = math.floor(len(ori_data) * 0.8)
train_x = np.array(data_x[0:train_end_index])
train_y = np.array(data_y[0:train_end_index])
val_x = np.array(data_x[train_end_index + 1:])
val_y = np.array(data_y[train_end_index + 1:])

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)

model = Sequential()
model.add(
    LSTM(
        units=512,
        return_sequences=False,
        # kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))
model.add(Dense(1, activation='tanh'))

model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mse')
# model.summary()
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
history = model.fit(train_x,
                    train_y,
                    validation_split=0.15,
                    epochs=512,
                    batch_size=32,
                    callbacks=[callback],
                    verbose=True)
pred_y_normalized = model.predict(val_x)
pred_y_ori = scalery.inverse_transform(pred_y_normalized)
pred_data_check = pd.DataFrame(pred_y_ori)
pred_data_check.to_csv('pred_y_check.csv')
print(pred_y_ori[-10:])
print(ori_data[-10:])