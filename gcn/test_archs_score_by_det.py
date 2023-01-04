import pickle
import time
from typing import Union, Tuple, List
import tensorflow as tf
import numpy as np
from model_util import calculate_dataset_level
from tensorflow.keras.datasets import cifar10
import os
import wget
from model_spec import ModelSpec
from scipy.stats import kendalltau


def setup():
    if not os.path.exists('nas-bench-101-data'):
        print('Downloading nas-bench-101-data...')
        file_name = wget.download('https://www.dropbox.com/s/vkexemlekfabxa1/nas-bench-101-data.zip?dl=1')
        print('Save data to {}'.format(file_name))
        os.system('unzip {}'.format(file_name))
        print(f'Unzip data finish.')


def get_balanced_n_data(n: int, num_class: int, x, y):
    assert  n % num_class == 0

    x_list = []
    y_list = []

    for i in range(num_class):
        cot = 0
        for data, label in zip(x, y):
            if label[0] == i:
                x_list.append(data)
                y_list.append(label)
                cot += 1
                if cot == n / num_class:
                    break

    return np.array(x_list), np.array(y_list)


def get_arch_score_and_acc_list(query_idx: int, sample_x: Union[tf.Tensor, np.ndarray]) -> Tuple[List, List]:
    score_list = []
    valid_acc_list = []
    input_shape = sample_x.shape

    for matrix_size in range(3, 4):
        with open(os.path.join('nas-bench-101-data', f'nasbench_101_cell_list_{matrix_size}.pkl'), 'rb') as f:
            cell_list = pickle.load(f)

        for cell in cell_list:
            score = calculate_dataset_level(ModelSpec(cell[0], cell[1]), input_shape, sample_x)
            val_acc = cell[2][query_idx]['validation_accuracy']
            print(score, val_acc)
            score_list.append(score)
            valid_acc_list.append(val_acc)

    return score_list, valid_acc_list


if __name__ == '__main__':
    setup()

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x, _ = get_balanced_n_data(20, 10, x_train, y_train)
    query_idx = 0

    score_list, val_acc_list = get_arch_score_and_acc_list(query_idx, x)
    kt, p = kendalltau(score_list, val_acc_list)
    print(kt, p)



