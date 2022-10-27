import sys

import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ECCConv, GINConvBatch, GATConv, GlobalSumPool, GlobalMaxPool, GlobalAvgPool, DiffPool
from spektral.data import BatchLoader
from tensorflow.python.keras.callbacks import CSVLogger

from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping

'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 ecc_conv (ECCConv)          multiple                  12544     
                                                                 
 batch_normalization (BatchN  multiple                 1024      
 ormalization)                                                   
                                                                 
 global_max_pool (GlobalMaxP  multiple                 0         
 ool)                                                            
                                                                 
 dropout (Dropout)           multiple                  0         
                                                                 
 dense (Dense)               multiple                  771       
                                                                 
=================================================================
Total params: 14,339
Trainable params: 13,827
Non-trainable params: 512
_________________________________________________________________
'''


class GNN_Model(Model):

    def __init__(self, n_hidden, mlp_hidden, activation: str, dropout=0.):
        super(GNN_Model, self).__init__()
        # self.graph_conv = ECCConv(n_hidden, activation=activation)
        self.graph_conv = GINConvBatch(n_hidden, mlp_hidden=mlp_hidden, mlp_activation=activation, mlp_batchnorm=True,
                                       activation=activation)
        # self.graph_conv = GATConv(n_hidden, attn_heads=1, activation=activation)
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalMaxPool()
        self.dropout = tensorflow.keras.layers.Dropout(dropout)
        self.dense = Dense(3)  # train_acc, valid_acc, test_acc

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out


def get_weighted_mse_loss_func(mid_point, alpha):
    def weighted_mse(y_true, y_pred):
        scale_mse_loss = K.switch((y_true < mid_point), alpha * tf.square(y_true - y_pred), tf.square(y_true - y_pred))
        return tf.reduce_mean(scale_mse_loss, axis=-1)

    return weighted_mse


if __name__ == '__main__':

    train_epochs = 100
    model_hidden = 256
    model_activation = 'relu'
    model_dropout = 0.2
    batch_size = 128
    weight_alpha = 436
    repeat = 1
    lr = 1e-3
    mlp_hidden = [128, 128, 256, 256]
    is_filtered = True
    patience = 20

    model = GNN_Model(n_hidden=model_hidden, mlp_hidden=mlp_hidden, activation=model_activation, dropout=model_dropout)

    # Set logger
    weight_filename = model.graph_conv.name + f'_filter{is_filtered}_a{weight_alpha}_r{repeat}_m{model_hidden}_b{batch_size}_dropout{model_dropout}_lr{lr}_mlp{mlp_hidden}'
    logging.basicConfig(filename=f'{weight_filename}.log', level=logging.INFO, force=True, filemode='w')
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)

    datasets = {
        'train': NasBench101Dataset(start=0, end=120000, preprocessed=is_filtered, repeat=repeat),
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
        datasets[key].apply(RemoveEdgeFeature_NasBench101())
        datasets[key].apply(SelectNoneNanData_NasBench101())
        logging.info(f'{datasets[key]}')

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile('adam', loss=get_weighted_mse_loss_func(mid_point=80, alpha=weight_alpha))

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
            if i[0] <= 80:
                logging.info(f'{i} {j}')
