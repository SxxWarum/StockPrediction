import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

f_open = open('stockdata\\test.csv')
ori_data = pd.read_csv(f_open)

# print(ori_data)

x, y = [], []
for i in range(len(ori_data) - 3):
    x.append(ori_data[i:i + 3, 0])
    y.append(ori_data[i + 2, 1])
train_x = np.array(x)
train_y = np.array(y)
print(train_x.shape)
print(train_y.shape)
