import csv
import os
import pickle
import numpy as np
from PIL import Image


def display_test():  # 读取测试集，预测，画图
    file_name = 'test.csv'
    file = open('NN.txt', 'rb')
    nn = pickle.load(file)
    count = 0
    len = 0
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        header_row = next(reader)
        print(header_row)

        for row in reader:
            len += 1
            img = np.array(row[1:785], dtype=np.uint8)
            img = img.reshape(28, 28)

            pre, _ = nn.predict(row[1: 785])
            if pre == int( row[0]):
                count += 1

    print('模型识别正确率：', count / len)

