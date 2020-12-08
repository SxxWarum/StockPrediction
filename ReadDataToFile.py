# git commit with no add
# VSC Test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import tushare as ts
import datetime
import os

HIDDEN_SIZE = 10
INPUT_SIZE = 8
OUTPUT_SIZE = 1
LR = 0.0001

def creat_windows():
    win = tk.Tk()  # 创建窗口
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    ww, wh = 800, 450
    x, y = (sw - ww) / 2, (sh - wh) / 2
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y - 40))  # 居中放置窗口

    win.title('获取股票数据，下方填股票代码')  # 窗口命名

    canvas = tk.Label(win)
    canvas.pack()

    L1 = tk.Label(win, text="获取股票数据，下方填股票代码，带.sz or .sh后缀")
    L1.pack()
    E1 = tk.Entry(win, bd=5)
    E1.pack()
    button1 = tk.Button(win, text="获取数据", command=lambda: getLable(E1))
    button1.pack()
    canvas.pack()
    win.mainloop()

def getLable(E1):
    string = E1.get()
    get_finacedata(string)

def get_finacedata(string):
    if string == '000001.SH' or string == '000300.SH':
        asset_string = 'I'
    else:
        asset_string = 'E'
    now = datetime.datetime.now().strftime('%Y%m%d')
    code_name = string
    #file_name = 'stockdata\\' + string[:-3] + '.csv'

    #print(now)
    ts.set_token('74978e36bfee58472b7277de21b9c781c2d9c6836f24e5e2188f2e99')
    api = ts.pro_api()
    data = ts.pro_bar(api=api,ts_code=code_name,freq='D',asset=asset_string,start_date='20010101',end_date=now)
    cols = list(data)
    #print(cols)
    #cols.insert(3, cols.pop(cols.index('open')))
    #print(cols)
    data = data.loc[:, cols].loc[::-1, :]
    # if os.path.exists(file_name):
    #     os.remove(file_name)
    # data.to_csv(file_name)
    basic_data = api.daily_basic(ts_code=code_name,start_date='20010101',end_date=now,fields='trade_date,turnover_rate_f,volume_ratio,pe_ttm,pb,ps_ttm,dv_ttm,free_share,circ_mv')
    cols = list(basic_data)
 
    basic_data = basic_data.loc[:, cols].loc[::-1, :]     
    data_all = pd.concat([data,basic_data],axis=1)
    #cols = list(data_all) 
    #data_all = data_all.loc[:, cols].loc[::-1, :]
    file_name = 'stockdata\\' + string[:-3] + '.csv'
    if os.path.exists(file_name):
        os.remove(file_name)
    data_all.to_csv(file_name)
    return 0  #存盘时，不返回数据
    #return data 不存盘时，返回data


def main():
    pass

if __name__ == "__main__":
    creat_windows()
