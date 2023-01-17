import logging
import pickle
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
from test_nasbench_ensemble import test_metric_partial_ensemble
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_MLP_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(64))
    model.add(layers.Dense(64))
    model.add(layers.Dense(64))
    model.add(layers.Dense(1))
    return model


def get_dataset_only_metric_order_to_ndarray(graph_dataset, metric: str, order: int) -> Dict[str, ndarray]:
    assert metric in ['flops', 'params']

    x_list = []
    y_list = []

    mean_std_dict = {'flops': [[1, 0],
                     [26638819.405752193, 65364232.74239259],
                     [4982109621333941.0, 1.9215401177096412e+16],
                     [1.3407101694764562e+24, 5.818062132947369e+24],
                     [3.940530586757471e+32, 1.759003441563639e+33],
                     [1.1822919313908888e+41, 5.313650616742194e+41],
                     [3.5647350741373663e+49, 1.6047937641856067e+50],
                     ],
                    'params': [[1, 0],
                                [90818.1944483277, 316351.1693478308],
                                [108326006790.59427, 693711124748.4598],
                                [2.1707989633206323e+17, 1.6338307382838333e+18],
                                [4.92969648346968e+23, 3.859175310803615e+24],
                                [1.1531006080581303e+30, 9.114026410774042e+30],
                                [2.716526562752633e+36, 2.152212202322697e+37]]
                }

    for data in graph_dataset:
        x = np.sum(data['x'], axis=0)
        x = np.take(x, [graph_dataset.features_dict[metric]])
        new_x = np.zeros((order))
        for i in range(1, order+1):
            new_x[i-1] = (x**i - mean_std_dict[metric][i][0]) / mean_std_dict[metric][i][1]

        y = data['y']
        x_list.append(new_x)
        y_list.append(y)

    return {'x': np.array(x_list), 'y': np.array(y_list)}


def train(model_output_dir: str, run: int, data_size: int, test_dataset: Dict[str, ndarray], batch_size: int, model_type: str, order: int, metric: str):
    is_filtered = True
    lr = 0.001
    train_epochs = 100
    patience = 20

    weight_file_dir = f'flops_order{order}_{model_type}_size{data_size}'
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

    for key in ['train', 'valid']:
        datasets[key].apply(RemoveTrainingTime_NasBench101())
        datasets[key].apply(LabelScale_NasBench101())
        datasets[key].apply(SelectNoneNanData_NasBench101())
        datasets[key].apply(Y_OnlyValidAcc())
        logging.info(f'key {datasets[key]}')
        datasets[key] = get_dataset_only_metric_order_to_ndarray(datasets[key], metric=metric, order=order)

    if model_type == 'mlp':
        model = get_MLP_model()
        model.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
        model.fit(datasets['train']['x'], datasets['train']['y'],
                  validation_data=(datasets['valid']['x'], datasets['valid']['y']),
                  epochs=train_epochs,
                  batch_size=batch_size,
                  callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),
                             CSVLogger(os.path.join(log_dir, log_dirs[1], f'{weight_filename}_history.log'))]
                  )
        model.save(weight_full_name)
    elif model_type == 'xgb':
        hp = {
            'n_estimators': 20000,
            'max_depth': 13,
            'min_child_weight': 39,
            'colsample_bylevel': 0.6909,
            'colsample_bytree': 0.2545,
            'reg_lambda': 31.3933,
            'reg_alpha': 0.2417,
            'learning_rate': 0.00824,
            'booster': 'gbtree',
            'early_stopping_rounds': 100,
            'random_state': 0,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'tree_method': 'gpu_hist'  # GPU
        }
        model = XGBRegressor(**hp)
        model.fit(X=datasets['train']['x'], y=datasets['train']['y'],
                  eval_set=[(datasets['valid']['x'], datasets['valid']['y'])])
        model.save_model(weight_full_name)
    elif model_type == 'lgb':
        hp = {
            'device': 'gpu',
            'n_estimators': 30000,  # equivalence to num_rounds
            'max_depth': 18,
            'num_leaves': 40,
            'max_bin': 255,
            'feature_fraction': 0.1532,
            'min_child_weight': 0.5822,
            'lambda_l1': 0.6909,
            'lambda_l2': 0.2545,
            'boosting_type': 'gbdt',
            'early_stopping_round': 100,
            'learning_rate': 0.0218,
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': 0
        }
        model = LGBMRegressor(**hp)
        model.fit(X=datasets['train']['x'], y=datasets['train']['y'],
                  eval_set=[(datasets['valid']['x'], datasets['valid']['y'])])
        pickle.dump(model, open(weight_full_name, 'wb'))
    else:
        raise ValueError(f'model_type {model_type} is not supported')

    logging.info(f'Model is saved to {weight_full_name}')

    pred = model.predict(datasets['test']['x'])
    loss = np.sqrt(mean_squared_error(datasets['test']['y'], pred))
    logging.info('Test MSE: {}'.format(loss))

    for y, predict in zip(datasets['test']['y'], pred):
        logging.info(f'{y} {predict}')

    return test_metric_partial_ensemble(os.path.join(log_dir, 'test_result'), weight_full_name, datasets['test'], model)


def train_n_runs(model_output_dir: str, n: int, data_size: int, test_dataset: Dict[str, ndarray], batch_size: int, model_type: str, order: int, metric: str):
    metrics = ['MSE', 'MAE', 'KT', 'P', 'mAP', 'NDCG']
    results = {i: [] for i in metrics}

    for i in range(n):
        # {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}
        metrics = train(model_output_dir, i, data_size, test_dataset, batch_size, model_type, order, metric)
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
    parser.add_argument('--model_type', type=str, default='mlp', help='Can be mlp, xgb, lgb')
    parser.add_argument('--train_by', type=str, default='flops', help='Can be flops, params')
    parser.add_argument('--select_range_list', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--order_list', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    for order in args.order_list:
        model_output_dir = f'flops_order{order}_{args.model_type}_model'
        Path(model_output_dir).mkdir(exist_ok=True)
        train_by = args.train_by
        # 194617
        test_dataset = NasBench101Dataset(start=174801, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True)
        test_dataset.apply(RemoveTrainingTime_NasBench101())
        test_dataset.apply(LabelScale_NasBench101())
        test_dataset.apply(SelectNoneNanData_NasBench101())
        test_dataset.apply(Y_OnlyValidAcc())
        test_dataset = get_dataset_only_metric_order_to_ndarray(test_dataset, metric=train_by, order=order)

        range_list = [
            [500, 10501, 500, 16],
            [11500, 20501, 1000, 256],
            [25500, 170501, 5000, 256]
        ]
        range_list = [range_list[i] for i in args.select_range_list]
        for r in range_list:
            for i in range(r[0], r[1], r[2]):
                train_n_runs(model_output_dir, n=10, data_size=i, test_dataset=test_dataset, batch_size=r[3], model_type=args.model_type, order=order, metric=train_by)

