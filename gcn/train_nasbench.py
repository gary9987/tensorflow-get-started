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

    def __init__(self, n_hidden):
        super().__init__()
        self.graph_conv = ECCConv(n_hidden, activation='relu')
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalSumPool()
        self.dense = Dense(3)  # train_acc, valid_acc, test_acc

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.pool(out)
        #out = self.dropout(out)
        out = self.dense(out)

        return out


def weighted_mse(y_true, y_pred):
    scale_mse_loss = K.switch((y_true < 80.0), 10 * tf.square(y_true - y_pred), tf.square(y_true - y_pred))
    return tf.reduce_mean(scale_mse_loss, axis=-1)


if __name__ == '__main__':

    logging.basicConfig(filename='train.log', level=logging.INFO, force=True)

    train_dataset = NasBench101Dataset(start=0, end=80000)
    valid_dataset = NasBench101Dataset(start=80001, end=160000)  # 80000 80250

    train_dataset.apply(NormalizeParAndFlop_NasBench101())
    valid_dataset.apply(NormalizeParAndFlop_NasBench101())

    train_dataset.apply(RemoveTrainingTime_NasBench101())
    valid_dataset.apply(RemoveTrainingTime_NasBench101())

    train_dataset.apply(SelectNoneNanData_NasBench101())
    valid_dataset.apply(SelectNoneNanData_NasBench101())

    train_dataset.apply(Normalize_x_10to15_NasBench101())
    valid_dataset.apply(Normalize_x_10to15_NasBench101())

    train_dataset.apply(NormalizeLayer_NasBench101())
    valid_dataset.apply(NormalizeLayer_NasBench101())

    train_dataset.apply(LabelScale_NasBench101())
    valid_dataset.apply(LabelScale_NasBench101())

    train_dataset.apply(NormalizeEdgeFeature_NasBench101())
    valid_dataset.apply(NormalizeEdgeFeature_NasBench101())

    print(train_dataset[0], valid_dataset)

    model = GNN_Model(n_hidden=128)
    #model.compile('adam', 'mean_squared_error', metrics=['mse'])
    model.compile('adam', loss=weighted_mse)

    batch_size = 128
    train_loader = BatchLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, epochs=100)

    valid_loader = BatchLoader(valid_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(valid_loader.load(), steps=valid_loader.steps_per_epoch)
    print('Test loss: {}'.format(loss))
    logging.info('Test loss: {}'.format(loss))
    # Test loss: [0.00380403408780694, 0.00380403408780694]

    valid_loader = BatchLoader(valid_dataset, batch_size=batch_size, shuffle=False, epochs=1)

    for data in valid_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            print(i, j)
            logging.info(f'{i} {j}')
