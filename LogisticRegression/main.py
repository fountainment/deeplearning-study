#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://www.kaggle.com/c/titanic


import csv
import numpy as np
import random


id_key = ['PassengerId']
y_key = ['Survived']
filter_out = ['Name', 'Ticket', 'Cabin'] + id_key + y_key
alpha = np.array([0.0005])


def sigmod(t):
    return 1.0 / (1.0 + np.exp(-t))


def data_filter(func, iter):
    return list(zip(*filter(func, list(zip(*iter)))))


def data_map(func, iter):
    return list(zip(*map(func, list(zip(*iter)))))


def data_func(data):
    data = list(data)
    if data[0] == 'Sex':
        for i in range(1, len(data)):
            data[i] = 1 if data[i] == 'male' else 0
    elif data[0] == 'Embarked':
        for i in range(1, len(data)):
            data[i] = {'':0, 'Q':1, 'S':2, 'C':3}[data[i]]
    for i in range(1, len(data)):
        if data[i] == '':
            data[i] = 0
        data[i] = float(data[i])
    return tuple(data)


def preprocess(raw_data):
    y_data = data_filter(lambda x : x[0] in y_key, raw_data)
    y_data = data_map(data_func, y_data)
    raw_data = data_filter(lambda x : x[0] not in filter_out, raw_data)
    raw_data = data_map(data_func, raw_data)
    keys = raw_data[0]
    data = raw_data[1:]
    y_data = y_data[1:]
    return [keys, data, y_data]


def train(train_data):
    keys = train_data[0]
    data = train_data[1]
    y_data = train_data[2]
    # train begin
    w = np.array([1.0 for i in range(len(keys))])
    b = np.array([1.0 for i in range(len(keys))])
    for i in range(10000):
        pack = list(zip(data, y_data))
        random.shuffle(pack)
        data, y_data = list(zip(*pack))
        for i in range(len(data)):
            y = np.array(y_data[i])
            x = np.array(data[i])
            z = w.T * x + b
            a = sigmod(z)
            dz = a - y
            dw = x * dz
            db = dz
            w -= alpha * dw
            b -= alpha * db
    # train end
    print(keys)
    print(w)
    print(b)
    return (w, b)


def test(test_data, args):
    keys = test_data[0]
    data = test_data[1]
    y_data = []
    w = args[0]
    b = args[1]
    # test begin
    for x in data:
        # Is this formula correct?
        result = sigmod(np.array(x).dot(w) + np.mean(b))
        result = 1 if result > 0.5 else 0
        y_data.append(result)
    # test end
    return y_data


def read_csv_file(file_name):
    return list(csv.reader(open(file_name, 'r')))


def write_csv_file(file_name, result):
    csv.writer(open(file_name, 'w', newline='\n')).writerows(result)


def main():
    raw_train_data = read_csv_file('train.csv')
    raw_test_data = read_csv_file('test.csv')
    train_data = preprocess(raw_train_data)
    test_data = preprocess(raw_test_data)
    train_result = train(train_data)
    test_result = test(test_data, train_result)
    test_id_start = 892
    test_csv_result = list(zip(id_key + list(range(test_id_start, test_id_start + len(test_result))), y_key + test_result))
    write_csv_file('result.csv', test_csv_result)


if __name__ == '__main__':
    main()
