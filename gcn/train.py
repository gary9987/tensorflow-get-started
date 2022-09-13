import pickle
from spektral.datasets import TUDataset

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ECCConv, GlobalSumPool
from spektral.data import BatchLoader
from learning_curve_dataset import LearningCurveDataset
import numpy as np
from transformation import *


class GNN_Model(Model):

    def __init__(self, n_hidden):
        super().__init__()
        self.graph_conv = ECCConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.3)
        self.dense = Dense(3)  # train_acc, valid_acc, test_acc

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)

        return out


if __name__ == '__main__':
    train_dataset = LearningCurveDataset(start=0, end=1000)
    valid_dataset = LearningCurveDataset(start=1001, end=2000)

    #transform1 = RemoveParAndFlopTransform()
    #train_dataset.apply(transform1)
    #valid_dataset.apply(transform1)

    #transform2 = OneHotToIndexTransform()
    #train_dataset.apply(transform2)
    #valid_dataset.apply(transform2)

    transform3 = NormalizeParAndFlopTransform()
    train_dataset.apply(transform3)
    valid_dataset.apply(transform3)

    print(train_dataset[0], valid_dataset)

    model = GNN_Model(n_hidden=128)
    model.compile('adam', 'mean_squared_error', metrics=['mse'])

    train_loader = BatchLoader(train_dataset, batch_size=128, shuffle=True)
    valid_loader = BatchLoader(valid_dataset, batch_size=256, shuffle=False)

    model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, epochs=150)

    loss = model.evaluate(valid_loader.load(), steps=valid_loader.steps_per_epoch)
    print('Test loss: {}'.format(loss))

    data = valid_loader.load().__next__()
    pred = model.predict(data[0])
    for i, j in zip(data[1], pred):
        print(i, j)

