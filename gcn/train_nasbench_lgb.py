import os.path
import sys
from argparse import ArgumentParser
from pathlib import Path
from keras import backend as K
from tensorflow.python.keras.callbacks import CSVLogger
from nas_bench_101_dataset import NasBench101Dataset, train_valid_test_split_dataset
from transformation import *
import logging
from test_nasbench_ensemble import test_metric_partial_ensemble
from nas_bench_101_dataset_partial import NasBench101DatasetPartial
from lightgbm import LGBMRegressor
import lightgbm
import pickle
from sklearn.metrics import mean_squared_error


def train(model_output_dir, run: int, data_size: int):
    hp = {
        'n_estimators': 30000, # equivalence to num_rounds
        'max_depth': 18,
        'num_leaves': 40,
        'max_bin': 336,
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
    is_filtered = True

    model = LGBMRegressor(**hp)

    weight_file_dir = f'lgb_size{data_size}'
    weight_filename = f'{weight_file_dir}_{run}'

    Path(os.path.join(model_output_dir, weight_file_dir)).mkdir(parents=True, exist_ok=True)
    weight_full_name = os.path.join(model_output_dir, weight_file_dir, weight_filename)

    print(weight_full_name)

    log_dir = f'{model_output_dir}_log'
    if not os.path.exists(os.path.join(log_dir, 'valid_log')):
        Path(os.path.join(log_dir, 'valid_log')).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, 'valid_log', f'{weight_filename}.log'), level=logging.INFO, force=True, filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    datasets = train_valid_test_split_dataset(NasBench101DatasetPartial(start=0, end=174800, size=data_size, matrix_size_list=[3, 4, 5, 6, 7],
                                                                        select_seed=run, preprocessed=is_filtered), ratio=[0.9, 0.1])
    datasets['test'] = NasBench101Dataset(start=174801, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=is_filtered)
    # 194617

    for key in datasets:
        datasets[key].apply(RemoveTrainingTime_NasBench101())
        datasets[key].apply(LabelScale_NasBench101())
        datasets[key].apply(RemoveEdgeFeature_NasBench101())
        datasets[key].apply(RemoveAllMetaData())
        datasets[key].apply(SelectNoneNanData_NasBench101())
        datasets[key].apply(Flatten4Ensemble_NasBench101())
        logging.info(f'key {datasets[key]}')

    for key, graphs in datasets.items():
        x_list = [np.squeeze(graph.a) for graph in graphs]
        y_list = [np.squeeze(graph.y[1]) for graph in graphs]
        datasets[key] = {'x': np.array(x_list), 'y': np.array(y_list)}

    model.fit(X=datasets['train']['x'], y=datasets['train']['y'], eval_set=[(datasets['valid']['x'], datasets['valid']['y'])])


    logging.info(f'Model will save to {weight_full_name}')
    pickle.dump(model, open(weight_full_name, 'wb'))

    pred = model.predict(datasets['test']['x'])
    loss = np.sqrt(mean_squared_error(datasets['test']['y'], pred))
    logging.info('Test MSE: {}'.format(loss))

    for y, predict in zip(datasets['test']['y'], pred):
        logging.info(f'{y} {predict}')

    return test_metric_partial_ensemble(os.path.join(log_dir, 'test_result'), weight_full_name, datasets['test'], model)


def train_n_runs(model_output_dir: str, n: int, data_size: int):
    metrics = ['MSE', 'MAE', 'KT', 'P', 'mAP', 'NDCG']
    results = {i: [] for i in metrics}

    for i in range(n):
        # {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}
        metrics = train(model_output_dir, i, data_size)
        print(metrics)
        for m in metrics:
            results[m].append(metrics[m])

        K.clear_session()

    logger = logging.getLogger('test_nasbench_ensemble')

    for key in results:
        logger.info(f'{key} mean: {sum(results[key])/len(results[key])}')
        logger.info(f'{key} min: {min(results[key])}')
        logger.info(f'{key} max: {max(results[key])}')
        logger.info(f'{key} std: {np.std(results[key])}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', type=str, default='ensemble_model_lgb')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Path(args.model_output_dir).mkdir(exist_ok=True)
    #train(args.model_output_dir, 0, 1000)
    range_list = [
        [500, 10501, 500],
        [11500, 20501, 1000],
        [25500, 170501, 5000]
    ]
    for r in range_list:
        for i in range(r[0], r[1], r[2]):
            train_n_runs(args.model_output_dir, n=10, data_size=i)
