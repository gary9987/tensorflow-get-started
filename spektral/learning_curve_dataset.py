from spektral.data import Dataset, DisjointLoader, Graph
import pickle
import numpy as np
import csv

class LearningCurveDataset(Dataset):
    def __init__(self, record_dic, record_dir, **kwargs):
        self.nodes = 67
        self.n_features = 7
        # 'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4, 'Classifier': 5, 'maxpool2x2': 6,
        self.ops_dict = {'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4}
        self.record_dir = record_dir
        self.record_dic = record_dic
        super().__init__(**kwargs)

    def read(self):
        graph_list = []

        for record in self.record_dic:
            matrix, ops, layers, log_file = np.array(record['matrix']), record['ops'], record['layers'], record['log_file']

            # Labels Y
            y = np.zeros(3)  # train_acc, valid_acc, test_acc
            with open(self.record_dir + log_file) as f:
                rows = csv.DictReader(f)
                tmp = []
                for row in rows:
                    tmp.append(row)
                y[0] = tmp[-1]['accuracy']
                y[1] = tmp[-1]['val_accuracy']
                y[2] = tmp[-1]['test_acc']

            # Node features X
            x = np.zeros((self.nodes, self.n_features), dtype=int)
            ops2idx = [self.ops_dict[op] for op in ops]
            for now_layer in range(11+1):
                if now_layer == 0:
                    x[now_layer][2] = 1  # stem is a 'conv3x3-bn-relu' type
                elif now_layer == 4:
                    x[22][6] = 1  # maxpool2x2
                elif now_layer == 8:
                    x[44][6] = 1  # maxpool2x2
                else:
                    now_group = now_layer // 4 + 1
                    node_start_no = now_group + 7 * (now_layer - now_group)
                    for i in range(len(ops2idx)):
                        x[i + node_start_no][ops2idx[i]] = 1
            x[66][5] = 1
            # Adjacency matrix A
            adj_matrix = np.zeros((self.nodes, self.nodes))
            '''
            0 convbn 128 
            1 cell    
            8 cell
            15 cell
            22 maxpool
            23 cell
            30 cell
            37 cell
            44 maxpool
            45 cell
            52 cell
            59 cell
            '''
            for now_layer in range(layers+1):
                if now_layer == 0:
                    adj_matrix[0][1] = 1  # stem to input node
                elif now_layer == 4:
                    adj_matrix[21][22] = 1  # output to maxpool
                    adj_matrix[22][23] = 1  # maxpool to input
                elif now_layer == 8:
                    adj_matrix[43][44] = 1  # output to maxpool
                    adj_matrix[44][45] = 1  # maxpool to input
                else:
                    now_group = now_layer // 4 + 1
                    node_start_no = now_group + 7 * (now_layer - now_group)
                    for i in range(matrix.shape[0]):
                        if i == 6:  # output node to next input node
                            adj_matrix[i + node_start_no][i + node_start_no + 1] = 1
                        else:
                            for j in range(matrix.shape[1]):
                                if matrix[i][j] == 1:
                                    adj_matrix[i + node_start_no][j + node_start_no] = 1





if __name__ == '__main__':
    file = open('../incremental/cifar10_log/cifar10.pkl', 'rb')
    record = pickle.load(file)
    file.close()

    dataset = LearningCurveDataset(record_dic=record, record_dir='../incremental/cifar10_log/')
    dataset.read()

