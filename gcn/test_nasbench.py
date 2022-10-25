import keras.models
import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ECCConv, GlobalSumPool, GlobalMaxPool, GlobalAvgPool
from spektral.data import BatchLoader
from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
import tensorflow as tf
import tensorflow.keras.backend as K


class GNN_Model(Model):

    def __init__(self, n_hidden, activation: str, dropout=0.):
        super(GNN_Model, self).__init__()
        self.graph_conv = ECCConv(n_hidden, activation=activation)
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
    logging.basicConfig(filename='test.log', level=logging.INFO, force=True)

    batch_size = 64
    weight_alpha = 1

    model = keras.models.load_model('weights', custom_objects={'weighted_mse': get_weighted_mse_loss_func(80, weight_alpha)})
    model.compile('adam', loss=get_weighted_mse_loss_func(mid_point=80, alpha=weight_alpha))

    valid_dataset = NasBench101Dataset(start=120001, end=160000, preprocessed=True)  # 80000 80250

    valid_dataset.apply(NormalizeParAndFlop_NasBench101())

    valid_dataset.apply(RemoveTrainingTime_NasBench101())

    valid_dataset.apply(Normalize_x_10to15_NasBench101())

    valid_dataset.apply(NormalizeLayer_NasBench101())

    valid_dataset.apply(LabelScale_NasBench101())

    valid_dataset.apply(NormalizeEdgeFeature_NasBench101())

    valid_dataset.apply(SelectNoneNanData_NasBench101())

    print(valid_dataset)

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