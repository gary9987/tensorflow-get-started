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
    '''
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

    if not os.path.exists('valid_log'):
        os.mkdir('valid_log')

    logging.basicConfig(filename=f'valid_log/{weight_filename}.log', level=logging.INFO, force=True, filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
    '''
    datasets = train_valid_test_split_dataset(NasBench101DatasetPartial(start=0, end=174800, size=data_size, matrix_size_list=[3, 4, 5, 6, 7],
                                                                        select_seed=run, preprocessed=True), ratio=[0.9, 0.1])
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

    model.fit(X=datasets['train']['x'], y=datasets['train']['y'], eval_set=[(datasets['valid']['x'], datasets['valid']['y'])], verbose=2)
    '''
    train_loader = BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    valid_loader = BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False)

    if not os.path.exists('learning_curve'):
        os.mkdir('learning_curve')
    history = model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch,
                        validation_data=valid_loader.load(), validation_steps=valid_loader.steps_per_epoch,
                        epochs=train_epochs,
                        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),
                                   CSVLogger(f"learning_curve/{weight_filename}_history.log")])

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

    return test_metric_partial('test_nasbench_partial_nochannel', weight_full_name, datasets['test'])
    '''


def train_n_runs(model_output_dir: str, n: int, data_size: int, no_channel=False):
    metrics = ['MSE', 'MAE', 'KT', 'P', 'mAP', 'NDCG']
    results = {i: [] for i in metrics}

    for i in range(n):
        # {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p}
        metrics = train(model_output_dir, i, data_size, no_channel=no_channel)
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
    train(args.model_output_dir, 0, 100000)
    #for i in range(500, 10501, 500):
    #    train_n_runs(args.model_output_dir, n=10, data_size=i, no_channel=True)
