import pickle
from spektral.datasets import TUDataset

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ECCConv, GlobalSumPool
from spektral.data import BatchLoader
from learning_curve_dataset import LearningCurveDataset


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
    file = open('../incremental/cifar10_log/cifar10.pkl', 'rb')
    record = pickle.load(file)
    file.close()

    train_dataset = LearningCurveDataset(record_dic=record, record_dir='../incremental/cifar10_log/', start=0,
                                         end=1999, inputs_shape=(None, 32, 32, 3), num_classes=10)
    valid_dataset = LearningCurveDataset(record_dic=record, record_dir='../incremental/cifar10_log/', start=2000,
                                         end=2999, inputs_shape=(None, 32, 32, 3), num_classes=10)
    print(train_dataset, valid_dataset)

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

