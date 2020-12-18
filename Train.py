import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score


def creat_windows():
    win = tk.Tk()  # 创建窗口
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    ww, wh = 400, 200
    x, y = (sw - ww) / 2, (sh - wh) / 2
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  # 居中放置窗口

    win.title('LSTM时间序列数据测试')  # 窗口命名

    # f_open = open('finacedata.csv')
    canvas = tk.Label(win)
    canvas.pack()

    global var
    var = tk.StringVar()  # 创建变量文字
    var.set('选择数据集')
    tk.Label(win,
             textvariable=var,
             bg='#C1FFC1',
             font=('宋体', 21),
             width=20,
             height=2).pack()

    canvas = tk.Label(win)
    L1 = tk.Label(win, text="填写代码")
    L1.pack()
    E1 = tk.Entry(win, bd=5)
    E1.pack()
    button1 = tk.Button(win, text="计算", command=lambda: Get_Ori_Data(E1))
    button1.pack()
    canvas.pack()
    win.mainloop()


def Get_Ori_Data(E1):
    global string
    string = E1.get()
    data_file_name = 'stockdata\\' + string + '.csv'
    f_open = open(data_file_name)
    df = pd.read_csv(f_open)  # 读入数据
    global ori_data
    ori_data = df.iloc[:, 3:].values
    main()


time_step = 30


def main():
    total_column_nums = 9
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_begin_index = 0
    train_end_index = math.floor(len(ori_data) * 0.8)
    test_end_index = math.floor(len(ori_data) * 0.9)
    train_data = np.array(ori_data[train_begin_index:train_end_index])
    normalized_train_data = scaler.fit_transform(train_data)
    #normalized_train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)  # 标准化
    test_data = np.array(ori_data[train_end_index + 1:test_end_index])
    normalized_test_data = scaler.fit_transform(test_data)
    #normalized_test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0)  # 标准化

    pred_data = np.array(ori_data[test_end_index + 1:])
    normalized_pred_data = scaler.fit_transform(pred_data)

    ckp_name = './checkpoints/train_model_3d.h5'
    switch = 'Train'

    if switch == 'Train':
        x, y = [], []
        for i in range(len(normalized_train_data) - time_step):
            x.append(normalized_train_data[i:i +
                                           time_step, :total_column_nums])
            y.append(normalized_train_data[i + time_step - 1,
                                           total_column_nums])
        data_train_X = np.array(x)
        data_train_Y = np.array(y)

        x1, y1 = [], []
        for i in range(len(normalized_test_data) - time_step):
            x1.append(normalized_test_data[i:i +
                                           time_step, :total_column_nums])
            y1.append(normalized_test_data[i + time_step - 1,
                                           total_column_nums])
        data_test_X = np.array(x1)
        data_test_Y = np.array(y1)

        model = Sequential()
        model.add(
            LSTM(units=100,
                 return_sequences=False,
                 input_shape=(data_train_X.shape[1], data_train_X.shape[2])))
        #model.add(LSTM(units=30,return_sequences=False)) # important
        # model.add(LSTM(units=100,return_sequences=True)) # important
        model.add(Dropout(0.5))
        model.add(Dense(9, activation='relu'))  #优化Step 6
        model.add(Dense(1, activation='relu'))  #优化Step 6
        # model.compile(optimizer=keras.optimizers.Adam(lr), loss = tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
        model.compile(optimizer=optimizers.Adam(lr=0.001), loss='mae')
        model.summary()

        history = model.fit(
            data_train_X,
            data_train_Y,
            epochs=50,
            batch_size=10,
            validation_data=(data_test_X, data_test_Y),
            #verbose=2,
            shuffle=False)
        model.save(ckp_name)
        test_y = model.predict(data_test_X)
        aa = np.zeros((test_data.shape[0], test_data.shape[1]))
        for i in range(test_y.shape[0]):
            aa[i + test_data.shape[0] - test_y.shape[0],
               total_column_nums] = test_y[i, 0]
        bb = scaler.inverse_transform(aa)
        for i in range(test_data.shape[0]):
            print("ori:", test_data[i, total_column_nums], "predict:",
                  bb[i, total_column_nums])
        plt.plot(test_data[:, total_column_nums],
                 color='red',
                 label='Original')
        plt.plot(bb[:, total_column_nums], color='green', label='Predict')
        plt.xlabel('the number of test data')
        plt.ylabel('earn_rate')
        plt.legend()
        plt.show()

    if switch == 'test':
        x, y = [], []
        for i in range(len(normalized_pred_data) - time_step):
            x.append(normalized_pred_data[i:i + time_step, :total_column_nums])
        data_pred_X = np.array(x)

        # model = Sequential()
        # model.add(LSTM(units=50, return_sequences=True,input_shape=(data_pred_X.shape[1], data_pred_X.shape[2])))
        # model.add(LSTM(units=50,return_sequences=True)) # important
        # #model.add(LSTM(units=50,return_sequences=True)) # important
        # model.add(Dropout(0.5))
        # model.add(Dense(1, activation='relu'))
        # # model.compile(optimizer=keras.optimizers.Adam(lr), loss = tf.losses.BinaryCrossentropy(),metrics=['accuracy'])
        # model.compile(optimizer='adam', loss = 'mae')
        model = tf.keras.models.load_model(ckp_name)

        pred_y = model.predict(data_pred_X)
        aa = np.zeros((pred_data.shape[0], pred_data.shape[1]))
        for i in range(pred_y.shape[0]):
            aa[i + pred_data.shape[0] - pred_y.shape[0],
               total_column_nums] = pred_y[i, 0]
        bb = scaler.inverse_transform(aa)
        for i in range(pred_data.shape[0]):
            print("ori:", pred_data[i, total_column_nums], "trans:",
                  bb[i, total_column_nums])
        ori_value = pred_data[:, total_column_nums]
        predict_value = bb[:, total_column_nums]
        print("ori_value shape:", tf.shape(ori_value))
        print("predict_value shape:", tf.shape(predict_value))
        # calculate MSE 均方误差
        mse = tf.losses.mean_squared_error(ori_value, predict_value)
        # calculate RMSE 均方根误差
        rmse = math.sqrt(tf.losses.mean_squared_error(ori_value,
                                                      predict_value))
        #calculate MAE 平均绝对误差
        mae = tf.losses.mean_absolute_error(ori_value, predict_value)
        #calculate R square
        r_square = r2_score(ori_value, predict_value)
        print('均方误差: %.6f' % mse)
        print('均方根误差: %.6f' % rmse)
        print('平均绝对误差: %.6f' % mae)
        print('R_square: %.6f' % r_square)
        plt.plot(ori_value, color='red', label='Original')
        plt.plot(predict_value, color='green', label='Predict')
        plt.xlabel('the number of test data')
        plt.ylabel('earn_rate')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    creat_windows()
