import sys
from spektral.data import BatchLoader
from tensorflow.python.keras.callbacks import CSVLogger
from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from nasbench_model import GNN_Model, get_weighted_mse_loss_func
from test_nasbench import test_method


if __name__ == '__main__':

    train_epochs = 100
    model_hidden = 256
    model_activation = 'relu'
    model_dropout = 0.2
    batch_size = 128
    weight_alpha = 5
    repeat = 80
    lr = 1e-3
    mid_point = 50
    mlp_hidden = [64, 64, 64, 64]
    #mlp_hidden = None
    is_filtered = True
    patience = 20

    model = GNN_Model(n_hidden=model_hidden, mlp_hidden=mlp_hidden, activation=model_activation, dropout=model_dropout)

    # Set logger
    if mlp_hidden is not None:
        weight_filename = model.graph_conv.name + f'_filter{is_filtered}_mp{mid_point}_a{weight_alpha}_r{repeat}_m{model_hidden}_b{batch_size}_dropout{model_dropout}_lr{lr}_mlp{tuple(mlp_hidden)}'
    else:
        weight_filename = model.graph_conv.name + f'_filter{is_filtered}_mp{mid_point}_a{weight_alpha}_r{repeat}_m{model_hidden}_b{batch_size}_dropout{model_dropout}_lr{lr}_mlp{mlp_hidden}'

    logging.basicConfig(filename=f'{weight_filename}.log', level=logging.INFO, force=True, filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    datasets = {
        'train': NasBench101Dataset(start=0, end=120000, preprocessed=is_filtered, repeat=repeat, mid_point=mid_point/100),
        'valid': NasBench101Dataset(start=120001, end=145000, preprocessed=is_filtered),
        'test': NasBench101Dataset(start=145001, end=169593, preprocessed=is_filtered),
    }

    for key in datasets:
        datasets[key].apply(NormalizeParAndFlop_NasBench101())
        datasets[key].apply(RemoveTrainingTime_NasBench101())
        datasets[key].apply(Normalize_x_10to15_NasBench101())
        datasets[key].apply(NormalizeLayer_NasBench101())
        datasets[key].apply(LabelScale_NasBench101())
        datasets[key].apply(NormalizeEdgeFeature_NasBench101())
        if 'ecc_con' not in weight_filename:
            datasets[key].apply(RemoveEdgeFeature_NasBench101())
        datasets[key].apply(SelectNoneNanData_NasBench101())
        logging.info(f'key {datasets[key]}')


    model.compile(tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=get_weighted_mse_loss_func(mid_point=mid_point, alpha=weight_alpha))

    train_loader = BatchLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    valid_loader = BatchLoader(datasets['valid'], batch_size=batch_size, shuffle=False)

    history = model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch,
                        validation_data=valid_loader.load(), validation_steps=valid_loader.steps_per_epoch,
                        epochs=train_epochs,
                        callbacks=[EarlyStopping(patience=patience, restore_best_weights=True),
                                   CSVLogger(f"{weight_filename}_history.log")])

    logging.info(f'{model.summary()}')

    logging.info(f'Model will save to {weight_filename}')
    model.save(weight_filename)

    test_loader = BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)
    # model.compile('adam', loss=get_weighted_mse_loss_func(mid_point=80, alpha=1))
    loss = model.evaluate(test_loader.load(), steps=valid_loader.steps_per_epoch)
    logging.info('Test loss: {}'.format(loss))

    test_loader = BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            logging.info(f'{i} {j}')

    test_loader = BatchLoader(datasets['test'], batch_size=batch_size, shuffle=False, epochs=1)
    logging.info('******************************************************************************')
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            if i[1] <= mid_point:
                logging.info(f'{i} {j}')

    test_method(weight_filename, mid_point)
