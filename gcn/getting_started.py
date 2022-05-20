from gcn.datasets import TUDataset
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from gcn.layers import GCNConv, GlobalSumPool
from gcn.data import BatchLoader


class MyFirstGNN(Model):

    def __init__(self, n_hidden, n_labels):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.5)
        self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        out = self.dense(out)

        return out


if __name__ == '__main__':
    dataset = TUDataset('PROTEINS')
    print(dataset)

    dataset.filter(lambda g: g.n_nodes < 500)
    print(dataset)

    model = MyFirstGNN(n_hidden=32, n_labels=dataset.n_labels)
    model.compile('adam', 'categorical_crossentropy')

    loader = BatchLoader(dataset, batch_size=32)

    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=10)

    print(model.summary())
