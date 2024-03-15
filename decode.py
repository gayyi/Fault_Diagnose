import linecache
import struct

import numpy as np
import torch
from matplotlib import pyplot as plt
import random

'''
只需调用此文件中的tim_to_data和get_sample函数。
前者需输入采集得到的同组dap和tim文件路径（默认第一通道采集），输出解码得到的实际数据（numpy array）
后者输入tim_to_data得到的array，并确定想从该array中得到的样本长度和数量，输出tensor数据，可直接作为模型inputs
'''

def get_txtdata(idx, channel, filedir):
    line = linecache.getline(filedir, idx+1)
    numbers = line.split()  # 通过空格分隔字符串得到数字列表
    float32_numbers = [np.float32(number) for number in numbers]  # 将数字转换为float32类型
    return float32_numbers[channel]


def get_txtlen(filedir):
    return sum(1 for _ in open(filedir))


# 从dap文件中获取线性变换的参数
def get_linear_arg(dap_pth, channel=1):
    '''将.dap文件读取进内存，首先定位内容为[LevelAndEU]的行，此行后面四行分别是四个通道的参数，即参数中的channel'''
    with open(dap_pth, 'r') as dap:
        lines = dap.readlines()
        tgt_str = '[LevelAndEU]'
        tgt_idx = 0
        for idx, line in enumerate(lines):
            if tgt_str in line:
                tgt_idx = idx+channel
        tgt_line = lines[tgt_idx]
        tgt_line.split(',')
        b, a = float(tgt_line.split(',')[3]), float(tgt_line.split(',')[4])

    return b, a


# 借助dap文件的参数，将.tim文件转为numpy array
def tim_to_data(file_pth, dap_pth):
    b, a = get_linear_arg(dap_pth)

    data_list = []
    with open(file_pth, 'rb') as file:
        while True:
            four_bytes = file.read(4)
            if not four_bytes:
                break
            integer = int.from_bytes(four_bytes, byteorder='little', signed=True)
            data_list.append((integer-b)/a)

    return np.array(data_list)


# 从tim_to_data得到的numpy arr中生成样本，输入参数分别为：输入的np arr，需要生成样本的数量(或batch size)
# 返回一个大小为(sp_num, 1, sp_len)的tensor
def get_sample(arr, sp_num, sp_len):
    sps = np.empty((sp_num, 1, sp_len))
    for i in range(sp_num):
        start = np.random.randint(0, len(arr)-sp_len)
        sp = arr[start:start+sp_len]
        sp = (sp / abs(sp).max()).reshape(1,sp_len)
        #sp = sp / abs(sp).max()
        sps[i] = sp
    return torch.tensor(sps, dtype=torch.float32)[:sp_len]  # ？？？


if __name__ == '__main__':
    tim_pth = 'strange_file/4号采集#1-1.tim'
    dap_pth = 'strange_file/4号采集.dap'
    txt_pth = 'strange_file/4号采集.txt'

    data = tim_to_data(tim_pth, dap_pth)

    ts = get_sample(data, 16, 32)
    print(ts)
    




