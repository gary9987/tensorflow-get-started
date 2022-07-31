import pickle

import tensorflow.keras.layers
from spektral.datasets import TUDataset

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ECCConv, GlobalSumPool, GlobalMaxPool, GlobalAvgPool
from spektral.data import BatchLoader
from learning_curve_dataset import LearningCurveDataset
from nas_bench_101_dataset import NasBench101Dataset
import numpy as np
from transformation import *


class GNN_Model(Model):

    def __init__(self, n_hidden):
        super().__init__()
        self.graph_conv = ECCConv(n_hidden)
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalAvgPool()
        self.dropout = Dropout(0.3)
        self.dense = Dense(3)  # train_acc, valid_acc, test_acc

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.dense(out)

        return out


if __name__ == '__main__':
    train_dataset = NasBench101Dataset(start=0, end=80000)
    valid_dataset = NasBench101Dataset(start=80001, end=160000)

    train_dataset.apply(NormalizeParAndFlopTransform_NasBench101())
    valid_dataset.apply(NormalizeParAndFlopTransform_NasBench101())

    train_dataset.apply(RemoveTrainingTime_NasBench101())
    valid_dataset.apply(RemoveTrainingTime_NasBench101())

    train_dataset.apply(SelectLabelQueryIdx_NasBench101(idx=0))
    valid_dataset.apply(SelectLabelQueryIdx_NasBench101(idx=1))

    train_dataset.apply(Normalize_x_10to15_NasBench101())
    valid_dataset.apply(Normalize_x_10to15_NasBench101())

    train_dataset.apply(NormalizeLayer_NasBench101())
    valid_dataset.apply(NormalizeLayer_NasBench101())

    train_dataset.apply(LabelScale_NasBench101())
    valid_dataset.apply(LabelScale_NasBench101())

    print(train_dataset[0], valid_dataset)

    model = GNN_Model(n_hidden=128)
    model.compile('adam', 'mean_squared_error', metrics=['mse'])

    train_loader = BatchLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = BatchLoader(valid_dataset, batch_size=128, shuffle=False)

    model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, epochs=100)

    loss = model.evaluate(valid_loader.load(), steps=valid_loader.steps_per_epoch)
    print('Test loss: {}'.format(loss))
    # Test loss: [0.00380403408780694, 0.00380403408780694]
    
    data = valid_loader.load().__next__()
    pred = model.predict(data[0])
    for i, j in zip(data[1], pred):
        print(i, j)

