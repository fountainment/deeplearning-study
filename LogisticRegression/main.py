#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import numpy as np

def preprocess(raw_data):
    pass

def train(data):
    pass

def test(data, args):
    pass

def read_csv_file(file_name):
    return list(csv.reader(open(file_name, 'r')))

def main():
    raw_train_data = read_csv_file('train.csv')
    raw_test_data = read_csv_file('test.csv')
    train_data = preprocess(raw_train_data)
    test_data = preprocess(raw_test_data)
    train_result = train(train_data)
    test_result = test(test_data, train_result)
    print(test_result)

if __name__ == '__main__':
    main()