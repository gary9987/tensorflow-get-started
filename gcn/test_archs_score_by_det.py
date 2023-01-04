import pickle
import random
import time
from typing import Union, Tuple, List
import tensorflow as tf
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from model_util import calculate_dataset_level
from tensorflow.keras.datasets import cifar10
import os
import wget
from model_spec import ModelSpec
from scipy.stats import kendalltau
from test_nasbench_metric import randon_select_data, mAP


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


def get_all_arch_list(shuffle=False):
    cell_list = []
    for matrix_size in range(3, 4):
        with open(os.path.join('nas-bench-101-data', f'nasbench_101_cell_list_{matrix_size}.pkl'), 'rb') as f:
            cell_list += pickle.load(f)

    if shuffle:
        random.shuffle(cell_list)

    return cell_list


def get_arch_score_and_acc_list(query_idx: int, sample_arch: List, sample_data: Union[tf.Tensor, np.ndarray]) -> Tuple[List, List]:
    score_list = []
    valid_acc_list = []
    input_shape = sample_data.shape

    for cell in tqdm(sample_arch):
        score = calculate_dataset_level(ModelSpec(cell[0], cell[1]), input_shape, sample_data)
        val_acc = cell[2][query_idx]['validation_accuracy']
        #print(f'score {score} val_acc {val_acc}')
        score_list.append(score)
        valid_acc_list.append(val_acc)

    return score_list, valid_acc_list


if __name__ == '__main__':
    setup()
    cell_list = get_all_arch_list(shuffle=True)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    sample_data, _ = get_balanced_n_data(20, 10, x_train, y_train)
    query_idx = 0

    num_select = 100
    test_count = 1000
    kt_list = []
    p_list = []
    mAP_list = []
    ndcg_list = []

    for i in range(test_count):
        print(f'Now testing {i} run')
        sample_arch = random.choices(cell_list, k=num_select)
        score_list, val_acc_list = get_arch_score_and_acc_list(query_idx, sample_arch, sample_data)
        kt, p = kendalltau(score_list, val_acc_list)
        kt_list.append(kt)
        p_list.append(p)
        mAP_list.append(mAP(score_list, val_acc_list, 0.1))
        ndcg_list.append(ndcg_score(np.asarray([score_list]), np.asarray([val_acc_list])))

    kt = sum(kt_list) / len(kt_list)
    p = sum(p_list) / len(p_list)
    avg_mAP = sum(mAP_list) / len(mAP_list)
    ndcg = sum(ndcg_list) / len(ndcg_list)
    print(f'Avg KT rank correlation: {kt}')
    print(f'Avg P value: {p}')
    print(f'Std KT rank correlation: {np.std(kt_list)}')
    print(f'Std P value: {np.std(p_list)}')
    print(f'Avg mAP value: {avg_mAP}')
    print(f'Std mAP value: {np.std(mAP_list)}')
    print(f'Avg ndcg value: {ndcg}')
    print(f'Std ndcg value: {np.std(ndcg_list)}')


