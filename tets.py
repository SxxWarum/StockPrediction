import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import math

f_open = open('stockdata\\test.csv')
ori_data = pd.read_csv(f_open)
ori_data_np = np.array(ori_data)

# print(ori_data)

x, y = [], []
for i in range(len(ori_data) - 3):
    x.append(ori_data_np[i:i + 3, :1])
    y.append(ori_data_np[i + 2, 1:])
data_x = np.array(x).astype(np.float32)
data_y = np.array(y).astype(np.float32)
train_begin_index = 0
train_end_index = math.floor(len(ori_data) * 0.8)
train_x = np.array(data_x[0:train_end_index]) / 20
train_y = np.array(data_y[0:train_end_index]) /20
val_x = np.array(data_x[train_end_index + 1:]) /20
val_y = np.array(data_y[train_end_index + 1:]) /20

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)


model = Sequential()
model.add(LSTM(units=512, return_sequences=False, input_shape=(train_x.shape[1], train_x.shape[2])) )
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mse')
# model.summary()

history = model.fit(train_x, train_y, validation_split=0.15, epochs=128, batch_size=32, verbose=True)
pred_y = model.predict(val_x)
print(pred_y)
print(val_y)