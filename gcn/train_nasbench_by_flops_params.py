import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import os
import tensorflow as tf
from numpy import ndarray
from tensorflow.keras import layers
from typing import Tuple, Dict
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping
from nas_bench_101_dataset_partial import NasBench101DatasetPartial
from nas_bench_101_dataset import NasBench101Dataset, train_valid_test_split_dataset
from transformation import *
from sklearn.metrics import mean_squared_error
from test_nasbench_rnn import test_metric_rnn


def get_MLP_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1))
    return model


def get_dataset_for_mlp(graph_dataset) -> Dict[str, ndarray]:
    x_list = []
    y_list = []

    for data in graph_dataset:
        x = np.sum(data['x'], axis=0)
        x = np.take(x, [graph_dataset.features_dict['flops'], graph_dataset.features_dict['params']])
        y = data['y']
        x_list.append(x)
        y_list.append(y)

    return {'x': np.array(x_list), 'y': np.array(y_list)}


def train(model_output_dir: str, run: int, data_size: int, test_dataset: Dict[str, ndarray], batch_size: int):
    is_filtered = True
    lr = 0.001
    train_epochs = 100
    patience = 20

    weight_file_dir = f'flops_params_mlp_size{data_size}'
    weight_filename = f'{weight_file_dir}_{run}'

    Path(os.path.join(model_output_dir, weight_file_dir)).mkdir(parents=True, exist_ok=True)
    weight_full_name = os.path.join(model_output_dir, weight_file_dir, weight_filename)

    print(weight_full_name)

    log_dir = f'{model_output_dir}_log'
    log_dirs = ['valid_log', 'learning_curve']
    for i in log_dirs:
        if not os.path.exists(os.path.join(log_dir, i)):
            Path(os.path.join(log_dir, i)).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, log_dirs[0], f'{weight_filename}.log'), level=logging.INFO, force=True, filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    datasets = train_valid_test_split_dataset(NasBench101DatasetPartial(start=0, end=174800, size=data_size, matrix_size_list=[3, 4, 5, 6, 7],
                                                                        select_seed=run, preprocessed=is_filtered), ratio=[0.9, 0.1])
    datasets['test'] = test_dataset
    # 194617

    for key in ['train', 'valid']:
        datasets[key].apply(NormalizeParAndFlop_NasBench101())
        datasets[key].apply(RemoveTrainingTime_NasBench101())
        datasets[key].apply(LabelScale_NasBench101())
        datasets[key].apply(SelectNoneNanData_NasBench101())
        datasets[key].apply(Y_OnlyValidAcc())
        logging.info(f'key {datasets[key]}')
        datasets[key] = get_dataset_for_mlp(datasets[key])

    model = get_MLP_model()
    model.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    model.fit(datasets['train']['x'], datasets['train']['y'],
              validation_data=(datasets['valid']['x'], datasets['valid']['y']),
              epochs=train_epochs,
              batch_size=batch_size,
              callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),
                         CSVLogger(os.path.join(log_dir, log_dirs[1], f'{weight_filename}_history.log'))]
              )

    logging.info(f'Model will save to {weight_full_name}')
    model.save(weight_full_name)

    pred = model.predict(datasets['test']['x'])
    loss = np.sqrt(mean_squared_error(datasets['test']['y'], pred))
    logging.info('Test MSE: {}'.format(loss))

    for y, predict in zip(datasets['test']['y'], pred):
        logging.info(f'{y} {predict}')

    return test_metric_rnn(os.path.join(log_dir, 'test_result'), weight_full_name, datasets['test'], model)


def train_n_runs(model_output_dir: str, n: int, data_size: int, test_dataset: Dict[str, ndarray], batch_size: int):
    metrics = ['MSE', 'MAE', 'KT', 'P', 'mAP', 'NDCG']
    results = {i: [] for i in metrics}

    for i in range(n):
        # {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}
        metrics = train(model_output_dir, i, data_size, test_dataset, batch_size)
        print(metrics)
        for m in metrics:
            results[m].append(metrics[m])

        tf.keras.backend.clear_session()

    logger = logging.getLogger('test_nasbench_rnn')

    for key in results:
        logger.info(f'{key} mean: {sum(results[key])/len(results[key])}')
        logger.info(f'{key} min: {min(results[key])}')
        logger.info(f'{key} max: {max(results[key])}')
        logger.info(f'{key} std: {np.std(results[key])}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', type=str, default='flops_params_mlp_model')
    parser.add_argument('--select_range_list', type=int, nargs='+', default=[0, 1, 2])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Path(args.model_output_dir).mkdir(exist_ok=True)

    # 194617
    test_dataset = NasBench101Dataset(start=174801, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True)
    test_dataset.apply(NormalizeParAndFlop_NasBench101())
    test_dataset.apply(RemoveTrainingTime_NasBench101())
    test_dataset.apply(LabelScale_NasBench101())
    test_dataset.apply(SelectNoneNanData_NasBench101())
    test_dataset.apply(Y_OnlyValidAcc())
    test_dataset = get_dataset_for_mlp(test_dataset)

    range_list = [
        [500, 10501, 500, 16],
        [11500, 20501, 1000, 256],
        [25500, 170501, 5000, 256]
    ]
    range_list = [range_list[i] for i in args.select_range_list]
    for r in range_list:
        for i in range(r[0], r[1], r[2]):
            train_n_runs(args.model_output_dir, n=10, data_size=i, test_dataset=test_dataset, batch_size=r[3])

