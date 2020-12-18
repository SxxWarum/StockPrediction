import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

f_open = open('stockdata\\test.csv')
ori_data = pd.read_csv(f_open)
ori_data_np = np.array(ori_data)

# print(ori_data)

x, y = [], []
for i in range(len(ori_data) - 3):
    x.append(ori_data_np[i:i + 3, :1])
    y.append(ori_data_np[i + 2, 1:])
train_x = np.array(x)
train_y = np.array(y)
print(train_x.shape)
print(train_y.shape)
