import os.path
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from spektral.data import BatchLoader
from tensorflow.python.keras.callbacks import CSVLogger
from nas_bench_101_dataset import NasBench101Dataset, train_valid_test_split_dataset
from transformation import *
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from nasbench_model import GNN_Model, get_weighted_mse_loss_func
from test_nasbench_partial import test_metric_partial
from nas_bench_101_dataset_partial import NasBench101DatasetPartial


no_channel = True
dataset_test = NasBench101Dataset(start=174801, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True)
dataset_test.apply(RemoveTrainingTime_NasBench101())
dataset_test.apply(Normalize_x_10to15_NasBench101())
dataset_test.apply(LabelScale_NasBench101())
dataset_test.apply(RemoveEdgeFeature_NasBench101())
if no_channel:
    dataset_test.apply(RemoveAllMetaData())
else:
    dataset_test.apply(RemoveMetaData())
dataset_test.apply(SelectNoneNanData_NasBench101())
dataset_test.apply(Y_OnlyValidAcc())


def train(model_output_dir, run: int, data_size: int, batch_size: int):
    train_epochs = 100
    model_hidden = 64
    model_activation = 'relu'
    model_dropout = 0.2
    weight_alpha = 1
    repeat = 1
    lr = 1e-3
    mlp_hidden = [64, 64, 64, 64]
    is_filtered = True
    patience = 20

    model = GNN_Model(n_hidden=model_hidden, mlp_hidden=mlp_hidden, activation=model_activation, dropout=model_dropout)

    # Set logger
    if mlp_hidden is not None:
        weight_file_dir = model.graph_conv.name + f'_filter{is_filtered}_a{weight_alpha}_size{data_size}_r{repeat}_m{model_hidden}_b{batch_size}_dropout{model_dropout}_lr{lr}_mlp{tuple(mlp_hidden)}'
        weight_filename = model.graph_conv.name + f'_filter{is_filtered}_a{weight_alpha}_size{data_size}_r{repeat}_m{model_hidden}_b{batch_size}_dropout{model_dropout}_lr{lr}_mlp{tuple(mlp_hidden)}_{run}'
    else:
        weight_file_dir = model.graph_conv.name + f'_filter{is_filtered}_a{weight_alpha}_size{data_size}_r{repeat}_m{model_hidden}_b{batch_size}_dropout{model_dropout}_lr{lr}_mlp{mlp_hidden}'
        weight_filename = model.graph_conv.name + f'_filter{is_filtered}_a{weight_alpha}_size{data_size}_r{repeat}_m{model_hidden}_b{batch_size}_dropout{model_dropout}_lr{lr}_mlp{mlp_hidden}_{run}'

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
                                                                        select_seed=run, preprocessed=True), ratio=[0.9, 0.1])
    datasets['test'] = dataset_test

    for key in ['train', 'valid']:
        datasets[key].apply(RemoveTrainingTime_NasBench101())
        datasets[key].apply(Normalize_x_10to15_NasBench101())
        datasets[key].apply(LabelScale_NasBench101())
        datasets[key].apply(RemoveEdgeFeature_NasBench101())
        if no_channel:
            datasets[key].apply(RemoveAllMetaData())
        else:
            datasets[key].apply(RemoveMetaData())
        datasets[key].apply(SelectNoneNanData_NasBench101())
        datasets[key].apply(Y_OnlyValidAcc())
        logging.info(f'key {datasets[key]}')


    #model.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss=get_weighted_mse_loss_func(mid_point=mid_point, alpha=weight_alpha))
    model.compile(tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

    train_loader = BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    valid_loader = BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False)

    if not os.path.exists('learning_curve'):
        os.mkdir('learning_curve')
    history = model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch,
                        validation_data=valid_loader.load(), validation_steps=valid_loader.steps_per_epoch,
                        epochs=train_epochs,
                        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),
                                   CSVLogger(os.path.join(log_dir, log_dirs[1], f'{weight_filename}_history.log'))])

    logging.info(f'{model.summary()}')

    logging.info(f'Model will save to {weight_full_name}')
    model.save(weight_full_name)

    test_loader = BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)
    # model.compile('adam', loss=get_weighted_mse_loss_func(mid_point=80, alpha=1))
    loss = model.evaluate(test_loader.load(), steps=valid_loader.steps_per_epoch)
    logging.info('Test loss: {}'.format(loss))

    test_loader = BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            logging.info(f'{i} {j}')

    return test_metric_partial(os.path.join(log_dir, 'test_result'), weight_full_name, datasets['test'])


def train_n_runs(model_output_dir: str, n: int, data_size: int, batch_size: int):
    metrics = ['MSE', 'MAE', 'KT', 'P', 'mAP', 'NDCG']
    results = {i: [] for i in metrics}

    for i in range(n):
        # {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}
        metrics = train(model_output_dir, i, data_size, batch_size)
        print(metrics)
        for m in metrics:
            results[m].append(metrics[m])

        tf.keras.backend.clear_session()

    logger = logging.getLogger('test_nasbench_partial')

    for key in results:
        logger.info(f'{key} mean: {sum(results[key])/len(results[key])}')
        logger.info(f'{key} min: {min(results[key])}')
        logger.info(f'{key} max: {max(results[key])}')
        logger.info(f'{key} std: {np.std(results[key])}')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model_output_dir', type=str, default='partial_model_nochannel')
    parser.add_argument('--select_range_list', type=int, nargs='+', default=[0])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    Path(args.model_output_dir).mkdir(exist_ok=True)
    range_list = [
        [500, 10501, 500, 16],
        [11500, 20501, 1000, 256],
        [25500, 170501, 5000, 256]
    ]
    range_list = [range_list[i] for i in args.select_range_list]
    #train(model_output_dir=args.model_output_dir)
    for r in range_list:
        for i in range(r[0], r[1], r[2]):
            train_n_runs(args.model_output_dir, n=10, data_size=i, batch_size=r[3])
