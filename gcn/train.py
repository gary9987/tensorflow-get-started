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
        self.dropout = Dropout(0.5)
        self.dense = Dense(3, activation='relu')  # train_acc, valid_acc, test_acc

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

    dataset = LearningCurveDataset(record_dic=record, record_dir='../incremental/cifar10_log/', num_samples=10000)
    print(dataset)
    #dataset = TUDataset('PROTEINS')

    model = GNN_Model(n_hidden=128)
    model.compile('adam', 'mean_squared_error', metrics=['accuracy'])

    loader = BatchLoader(dataset, batch_size=64, shuffle=False)

    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=100)

    print(model.summary())
