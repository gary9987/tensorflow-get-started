import os.path
from pathlib import Path
import keras.models
from spektral.data import BatchLoader
from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
from nasbench_model import is_weight_dir
from scipy.stats import kendalltau
import tensorflow as tf
from test_nasbench_metric import randon_select_data


def test_metric_partial(log_dir, weight_path):

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = os.path.join(log_dir, f'{Path(weight_path).name}_test.log')

    if os.path.exists(log_path):
        os.remove(log_path)

    if not os.path.exists(Path(log_path).parents[0]):
        Path(log_path).parents[0].mkdir()

    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)
    batch_size = 64

    #model = keras.models.load_model(weight_path,
    #                                custom_objects={'weighted_mse': tf.keras.losses.MeanSquaredError()})
    model = keras.models.load_model(weight_path)

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
    mse = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test MSE loss: {}'.format(mse))
    logging.info('Test MSE loss: {}'.format(mse))

    model.compile('adam', loss='mae')

    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    mae = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test MAE loss: {}'.format(mae))
    logging.info('Test MAE loss: {}'.format(mae))

    label_array = np.array([])
    pred_array = np.array([])

    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            # logging.info(f'{i} {j}')
            valid_label, valid_predict = i[1], j[1]
            label_array = np.concatenate((label_array, np.array(valid_label)), axis=None)
            pred_array = np.concatenate((pred_array, np.array(valid_predict)), axis=None)

    num_select = 100
    test_count = 1000
    kt_list = []
    p_list = []

    for _ in range(test_count):
        pred_list, label_list = randon_select_data(pred_array, label_array, 0, num_select, 0)
        kt, p = kendalltau(pred_list, label_list)
        kt_list.append(kt)
        p_list.append(p)

    kt = sum(kt_list) / len(kt_list)
    p = sum(p_list) / len(p_list)
    logging.info(f'Avg KT rank correlation: {kt}')
    logging.info(f'Avg P value: {p}')
    logging.info(f'Std KT rank correlation: {np.std(kt_list)}')
    logging.info(f'Std P value: {np.std(p_list)}')

    return {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}

if __name__ == '__main__':
    log_dir = 'test_partial_model'
    model_dir = 'partial_model'

    for filename in os.listdir():
        if os.path.isdir(filename) and is_weight_dir(filename):
            print(f'Now test {filename}')
            test_metric_partial(log_dir, filename)

