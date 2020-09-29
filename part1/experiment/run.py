#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lstmcount.py
# Author: Qian Ge <geqian1001@gmail.com>

import sys
import numpy as np

sys.path.append('../')
# import assign as assign
from src.lstm import LSTMcell
from src.count import count_0_in_seq


def one_hot(inputs, depth):
    n_inputs = len(inputs)
    one_hot_vector = np.zeros((n_inputs, depth))
    one_hot_vector[np.arange(n_inputs), inputs] = 1
    return one_hot_vector

if __name__ == "__main__":
    o_input_seq = [1, 1, 0, 4, 3, 4, 0, 2, 0, 2, 0, 4, 3, 0, 2, 4, 5, 0, 9, 0, 4]
    input_seq = one_hot(o_input_seq, depth=10)
    print('Input sequence: {}'.format(o_input_seq))

    count_num = count_0_in_seq(input_seq, count_type='task1')
    print('Number of 0: {}'.format(count_num))

    count_num = count_0_in_seq(input_seq, count_type='task2')
    print('Number of 0 after the first 2: {}'.format(count_num))

    count_num = count_0_in_seq(input_seq, count_type='task3')
    print('Number of 0 after 2 but erase by 3: {}'.format(count_num))