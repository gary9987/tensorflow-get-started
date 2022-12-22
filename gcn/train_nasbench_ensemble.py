import os.path
import sys
from argparse import ArgumentParser
from pathlib import Path
from keras import backend as K
from tensorflow.python.keras.callbacks import CSVLogger
from nas_bench_101_dataset import NasBench101Dataset, train_valid_test_split_dataset
from transformation import *
import logging
from test_nasbench_partial import test_metric_partial
from nas_bench_101_dataset_partial import NasBench101DatasetPartial
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def train(model_output_dir, run: int, data_size: int):
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
        'tree_method': 'gpu_hist' # GPU
    }
    is_filtered = True

    model = XGBRegressor(**hp)

    weight_file_dir = f'xgb_size{data_size}'
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
        y_list = [np.squeeze(graph.y) for graph in graphs]
        datasets[key] = {'x': np.array(x_list), 'y': np.array(y_list)}

    model.fit(X=datasets['train']['x'], y=datasets['train']['y'], eval_set=[(datasets['valid']['x'], datasets['valid']['y'])])


    logging.info(f'Model will save to {weight_full_name}')
    model.save_model(weight_full_name)

    pred = model.predict(datasets['test']['x'])
    loss = np.sqrt(mean_squared_error(datasets['test']['y'], pred))
    logging.info('Test MSE: {}'.format(loss))

    for y, predict in zip(datasets['test']['y'], pred):
        logging.info(f'{y} {predict}')

    return test_metric_partial('test_nasbench_partial_nochannel', weight_full_name, datasets['test'])



def train_n_runs(model_output_dir: str, n: int, data_size: int, no_channel=False):
    metrics = ['MSE', 'MAE', 'KT', 'P', 'mAP', 'NDCG']
    results = {i: [] for i in metrics}

    for i in range(n):
        # {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}
        metrics = train(model_output_dir, i, data_size)
        print(metrics)
        for m in metrics:
            results[m].append(metrics[m])

        K.clear_session()

    logger = logging.getLogger('test_nasbench_partial')

    for key in results:
        logger.info(f'{key} mean: {sum(results[key])/len(results[key])}')
        logger.info(f'{key} min: {min(results[key])}')
        logger.info(f'{key} max: {max(results[key])}')
        logger.info(f'{key} std: {np.std(results[key])}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', type=str, default='ensemble_model')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Path(args.model_output_dir).mkdir(exist_ok=True)
    train(args.model_output_dir, 0, 1000)
    #for i in range(500, 10501, 500):
    #    train_n_runs(args.model_output_dir, n=10, data_size=i, no_channel=True)
