import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ECCConv, GINConvBatch, GlobalSumPool, GlobalMaxPool, GlobalAvgPool
from spektral.data import BatchLoader
from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
import tensorflow as tf
import tensorflow.keras.backend as K

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
        #self.graph_conv = ECCConv(n_hidden, activation=activation)
        self.graph_conv = GINConvBatch(n_hidden, mlp_hidden=mlp_hidden, mlp_activation=activation, mlp_batchnorm=True, activation=activation)
        #self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalMaxPool()
        self.dropout = tensorflow.keras.layers.Dropout(dropout)
        self.dense = Dense(3)  # train_acc, valid_acc, test_acc

    def call(self, inputs):
        out = self.graph_conv(inputs)
        #out = self.bn(out)
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
    logging.basicConfig(filename='train.log', level=logging.INFO, force=True)

    train_epochs = 100
    model_hidden = 256
    model_activation = 'relu'
    model_dropout = 0.2
    batch_size = 64
    weight_alpha = 1.2
    repeat = 436
    lr = 1e-3
    mlp_hidden = [128, 512, 2048]

    train_dataset = NasBench101Dataset(start=0, end=120000, preprocessed=True, repeat=repeat)
    valid_dataset = NasBench101Dataset(start=120001, end=160000, preprocessed=True)  # 80000 80250
    #train_dataset = NasBench101Dataset(start=0, end=100)
    #valid_dataset = NasBench101Dataset(start=0, end=100)  # 80000 80250

    train_dataset.apply(NormalizeParAndFlop_NasBench101())
    valid_dataset.apply(NormalizeParAndFlop_NasBench101())

    train_dataset.apply(RemoveTrainingTime_NasBench101())
    valid_dataset.apply(RemoveTrainingTime_NasBench101())

    train_dataset.apply(Normalize_x_10to15_NasBench101())
    valid_dataset.apply(Normalize_x_10to15_NasBench101())

    train_dataset.apply(NormalizeLayer_NasBench101())
    valid_dataset.apply(NormalizeLayer_NasBench101())

    train_dataset.apply(LabelScale_NasBench101())
    valid_dataset.apply(LabelScale_NasBench101())

    train_dataset.apply(NormalizeEdgeFeature_NasBench101())
    valid_dataset.apply(NormalizeEdgeFeature_NasBench101())

    train_dataset.apply(RemoveEdgeFeature_NasBench101())
    valid_dataset.apply(RemoveEdgeFeature_NasBench101())

    train_dataset.apply(SelectNoneNanData_NasBench101())
    valid_dataset.apply(SelectNoneNanData_NasBench101())

    print(train_dataset, valid_dataset)

    model = GNN_Model(n_hidden=model_hidden, mlp_hidden=mlp_hidden, activation=model_activation, dropout=model_dropout)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile('adam', loss=get_weighted_mse_loss_func(mid_point=80, alpha=weight_alpha))

    train_loader = BatchLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, epochs=train_epochs)
    print(model.summary())
    model.save('weights')

    valid_loader = BatchLoader(valid_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(valid_loader.load(), steps=valid_loader.steps_per_epoch)
    print('Test loss: {}'.format(loss))
    logging.info('Test loss: {}'.format(loss))
    # Test loss: [0.00380403408780694, 0.00380403408780694]

    valid_loader = BatchLoader(valid_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    for data in valid_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            logging.info(f'{i} {j}')

    valid_loader = BatchLoader(valid_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    logging.info('******************************************************************************')
    for data in valid_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            if i[0] <= 80:
                logging.info(f'{i} {j}')
