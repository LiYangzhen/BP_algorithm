from BPNeuralNetwork import NeuralNetwork
import numpy as np
import pickle
import csv


def train():
    file_name = 'train.csv'
    y = []
    x = []
    y_t = []
    x_t = []
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        header_row = next(reader)
        print(header_row)
        for row in reader:
            if np.random.random() < 0.8:  # 80%的数据用于训练
                y.append(int(row[0]))
                x.append(list(map(int, row[1:])))
            else:
                y_t.append(int(row[0]))
                x_t.append(list(map(int, row[1:])))
    len_train = len(y)
    len_test = len(y_t)
    print('训练集大小%d，测试集大小%d' % (len_train, len_test))
    x = np.array(x)
    y = np.array(y)
    nn = NeuralNetwork([784, 300, 12])  # 神经网络各层神经元个数
    nn.fit(x, y, 0.15, 100)
    file = open('NN.txt', 'wb')
    pickle.dump(nn, file)
    count = 0
    for i in range(len_test):
        p, _ = nn.predict(x_t[i])
        print(p, y_t[i])
        if p == y_t[i]:
            count += 1
    print('模型识别正确率：', count / len_test)


train()