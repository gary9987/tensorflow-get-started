import os.path
from pathlib import Path

import keras.models
import numpy as np
from spektral.data import BatchLoader
from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
from nasbench_model import get_weighted_mse_loss_func
from test_nasbench import is_weight_dir
import random


def is_misjudgment(label, predict, mid_point: int, num_select: int, num_minor: int, num_judge: int):
    assert num_select > num_minor
    assert num_judge <= num_select

    num_major = num_select - num_minor

    pred_list = []
    label_list = []

    for select_type, num in zip(['major', 'minor'], [num_major, num_minor]):
        for _ in range(num):
            rand_idx = random.randint(0, label.shape[0]-1)
            if select_type == 'major':
                while label[rand_idx] <= mid_point:
                    rand_idx = random.randint(0, label.shape[0] - 1)
            elif select_type == 'minor':
                while label[rand_idx] > mid_point:
                    rand_idx = random.randint(0, label.shape[0] - 1)

            pred_list.append(predict[rand_idx])
            label_list.append(label[rand_idx])

    sorted_idx = sorted(range(len(pred_list)), key=lambda k: pred_list[k])

    # Return True when the minor data appears in the top num_judge data
    for idx in sorted_idx[num_select-num_judge:]:
        if label_list[idx] <= mid_point:
            return True


def test_metric(weight_path, mid_point):
    log_path = f'test_metric/{weight_path}_test_metric.log'

    if os.path.exists(log_path):
        os.remove(log_path)

    if not os.path.exists(Path(log_path).parents[0]):
        Path(log_path).parents[0].mkdir()

    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)
    batch_size = 64
    weight_alpha = 1

    model = keras.models.load_model(weight_path,
                                    custom_objects={'weighted_mse': get_weighted_mse_loss_func(mid_point, weight_alpha)})

    test_dataset = NasBench101Dataset(start=174801, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True)
    print(test_dataset)

    test_dataset.apply(NormalizeParAndFlop_NasBench101())
    test_dataset.apply(RemoveTrainingTime_NasBench101())
    test_dataset.apply(Normalize_x_10to15_NasBench101())
    test_dataset.apply(NormalizeLayer_NasBench101())
    test_dataset.apply(LabelScale_NasBench101())
    test_dataset.apply(NormalizeEdgeFeature_NasBench101())
    if 'ecc_con' not in weight_path:
        test_dataset.apply(RemoveEdgeFeature_NasBench101())
    test_dataset.apply(SelectNoneNanData_NasBench101())

    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)

    label_array = np.array([])
    pred_array = np.array([])

    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            # logging.info(f'{i} {j}')
            valid_label, valid_predict = i[1], j[1]

            label_array = np.concatenate((label_array, np.array(valid_label)), axis=None)
            pred_array = np.concatenate((pred_array, np.array(valid_predict)), axis=None)

    test_count = 100
    mis_count = 0
    for _ in range(test_count):
        if is_misjudgment(pred_array, label_array, mid_point, 100, 1, 50):
            mis_count += 1

    logging.info(f'The misjudgement ratio: {(mis_count/test_count)*100}%')


if __name__ == '__main__':
    '''
    for filename in os.listdir():
        if os.path.isdir(filename) and is_weight_dir(filename):
            print(f'Now test {filename}')
            test_method(filename)
    
    for i in range(10, 91, 10):
        print(f'Now mp is {i}')
    '''
    test_metric('gin_conv_batch_filterTrue_mp50_a1_r1_m256_b128_dropout0.2_lr0.001_mlp(64, 64, 64, 64)', 50)
