from spektral.data import Dataset, DisjointLoader, Graph
import pickle
import numpy as np


class LearningCurveDataset(Dataset):
    def __init__(self, record_dic, **kwargs):
        self.nodes = 67
        self.feats = 5
        self.record_dic = record_dic
        super().__init__(**kwargs)

    def read(self):
        graph_list = []
        for record in self.record_dic:
            matrix, ops, layers = np.array(record['matrix']), record['ops'], record['layers']
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

            print(matrix)
            print(adj_matrix)
            pass


if __name__ == '__main__':
    file = open('../incremental/cifar10_log/cifar10.pkl', 'rb')
    record = pickle.load(file)
    file.close()

    dataset = LearningCurveDataset(record_dic=record)
    dataset.read()

