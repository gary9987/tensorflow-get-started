import logging
import sys
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import os
import tensorflow as tf
from numpy import ndarray
from spektral.data import Dataset
from tensorflow.keras import layers
from typing import Tuple, Dict
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping
from gcn.nas_bench_101_dataset_partial import NasBench101DatasetPartial
from nas_bench_101_dataset import NasBench101Dataset, train_valid_test_split_dataset
from transformation import *
from sklearn.metrics import mean_squared_error
from test_nasbench_rnn import test_metric_rnn


def bfs(matrix):
    todo = [1]
    visit = [False] * 8
    visit[1] = True
    order = []

    num_nodes = 7
    while len(todo) != 0:
        r = len(todo)
        for i in range(r):
            front = todo.pop(0)
            order.append(front)
            for j in range(1, num_nodes + 1):
                if matrix[front][j] != 0 and visit[j] is not True:
                    todo.append(j)
                    visit[j] = True

    return order


def get_dataset(graph_dataset: Dataset) -> Dict[str, ndarray]:
    x_list = []
    y_list = []

    for data in graph_dataset:
        x = data['x']
        y = data['y']
        a = data['a']

        order = bfs(a)
        out = np.zeros((a.shape[0], x.shape[1]), dtype=float)

        now_idx = 0
        for now_layer in range(12):
            if now_layer == 0:
                out[now_idx] = x[0]
                now_idx += 1
            elif now_layer == 4:
                out[now_idx] = x[22]
                now_idx += 1
            elif now_layer == 8:
                out[now_idx] = x[44]
                now_idx += 1
            else:
                now_group = now_layer // 4 + 1
                node_start_no = now_group + 7 * (now_layer - now_group)

                for i in range(len(order)):
                    real_idx = node_start_no + order[i] - 1
                    out[now_idx] = x[real_idx]
                    now_idx += 1

        x_list.append(out)
        y_list.append(y)

    return {'x': np.array(x_list), 'y': np.array(y_list)}


def get_LSTM_model(units: int, shape: Tuple[int, int]):
    model = tf.keras.Sequential()
    #model.add(layers.LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(layers.LSTM(units, input_shape=(shape[0], shape[1])))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1))
    return model


def train(model_output_dir: str, run: int, data_size: int, test_dataset: Dict[str, ndarray], batch_size: int, rm_all_metadata: bool):
    units = 128
    is_filtered = True
    lr = 0.001
    train_epochs = 100
    patience = 20

    weight_file_dir = f'lstm_size{data_size}'
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
        datasets[key].apply(Normalize_x_10to15_NasBench101())
        datasets[key].apply(NormalizeLayer_NasBench101())
        datasets[key].apply(LabelScale_NasBench101())
        datasets[key].apply(SelectNoneNanData_NasBench101())
        datasets[key].apply(Y_OnlyValidAcc())
        if rm_all_metadata:
            datasets[key].apply(RemoveAllMetaData())
        logging.info(f'key {datasets[key]}')
        datasets[key] = get_dataset(datasets[key])

    model = get_LSTM_model(units, (datasets['train']['x'].shape[1], datasets['train']['x'].shape[2]))
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


def train_n_runs(model_output_dir: str, n: int, data_size: int, test_dataset: Dict[str, ndarray], batch_size: int, rm_all_metadata: bool):
    metrics = ['MSE', 'MAE', 'KT', 'P', 'mAP', 'NDCG']
    results = {i: [] for i in metrics}

    for i in range(n):
        # {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}
        metrics = train(model_output_dir, i, data_size, test_dataset, batch_size, rm_all_metadata)
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
    parser.add_argument('--model_output_dir', type=str, default='lstm_model')
    parser.add_argument('--select_range_list', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--rm_all_metadata', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Path(args.model_output_dir).mkdir(exist_ok=True)

    # 194617
    test_dataset = NasBench101Dataset(start=174801, end=174810, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True)
    test_dataset.apply(NormalizeParAndFlop_NasBench101())
    test_dataset.apply(RemoveTrainingTime_NasBench101())
    test_dataset.apply(Normalize_x_10to15_NasBench101())
    test_dataset.apply(NormalizeLayer_NasBench101())
    test_dataset.apply(LabelScale_NasBench101())
    test_dataset.apply(SelectNoneNanData_NasBench101())
    test_dataset.apply(Y_OnlyValidAcc())
    if args.rm_all_metadata:
        test_dataset.apply(RemoveAllMetaData())
    test_dataset = get_dataset(test_dataset)

    range_list = [
        [500, 10501, 500, 16],
        [11500, 20501, 1000, 256],
        [25500, 170501, 5000, 256]
    ]
    range_list = [range_list[i] for i in args.select_range_list]
    for r in range_list:
        for i in range(r[0], r[1], r[2]):
            train_n_runs(args.model_output_dir, n=10, data_size=i, test_dataset=test_dataset, batch_size=r[3], rm_all_metadata=args.rm_all_metadata)

