from spektral.data import Dataset, Graph
import pickle
import numpy as np
import csv
from model_spec import ModelSpec


def compute_vertex_channels(input_channels, output_channels, matrix):
    """
    Computes the number of channels at every vertex.
    Given the input channels and output channels, this calculates the number of
    channels at each interior vertex. Interior vertices have the same number of
    channels as the max of the channels of the vertices it feeds into. The output
    channels are divided amongst the vertices that are directly connected to it.
    When the division is not even, some vertices may receive an extra channel to
    compensate.
    Args:
        input_channels: input channel count.
        output_channels: output channel count.
        matrix: adjacency matrix for the module (pruned by model_spec).
    Returns:
        list of channel counts, in order of the vertices.
    """
    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = input_channels
    vertex_channels[num_vertices - 1] = output_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = output_channels // in_degree[num_vertices - 1]
    correction = output_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            vertex_channels[v] = interior_channels
            if correction:
                vertex_channels[v] += 1
                correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # tf.logging.info('vertex_channels: %s', str(vertex_channels))

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == output_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return vertex_channels


class LearningCurveDataset(Dataset):
    def __init__(self, record_dic, record_dir, start, end, **kwargs):
        self.nodes = 67
        self.n_features = 7
        self.start = start
        self.end = end
        # 'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4, 'Classifier': 5,
        # 'maxpool2x2': 6,
        self.ops_dict = {'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4}
        self.record_dir = record_dir
        self.record_dic = record_dic
        super().__init__(**kwargs)

    def read(self):
        graph_list = []

        for record in self.record_dic[self.start: self.end + 1]:
            matrix, ops, layers, log_file = np.array(record['matrix']), record['ops'], record['layers'], record[
                'log_file']

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
            x = np.zeros((self.nodes, self.n_features), dtype=float)
            ops2idx = [self.ops_dict[op] for op in ops]
            for now_layer in range(11 + 1):
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
            adj_matrix = np.zeros((self.nodes, self.nodes), dtype=float)
            
            #0 convbn 128 
            #1 cell    
            #8 cell
            #15 cell
            #22 maxpool
            #23 cell
            #30 cell
            #37 cell
            #44 maxpool
            #45 cell
            #52 cell
            #59 cell
            
            for now_layer in range(layers + 1):
                if now_layer == 0:
                    if now_layer == layers:
                        adj_matrix[0][self.nodes - 1] = 1  # to classifier
                    else:
                        adj_matrix[0][1] = 1  # stem to input node
                elif now_layer == 4:
                    adj_matrix[21][22] = 1  # output to maxpool
                    if now_layer == layers:
                        adj_matrix[22][self.nodes - 1] = 1  # to classifier
                    else:
                        adj_matrix[22][23] = 1  # maxpool to input
                elif now_layer == 8:
                    adj_matrix[43][44] = 1  # output to maxpool
                    if now_layer == layers:
                        adj_matrix[44][self.nodes - 1] = 1  # to classifier
                    else:
                        adj_matrix[44][45] = 1  # maxpool to input
                else:
                    now_group = now_layer // 4 + 1
                    node_start_no = now_group + 7 * (now_layer - now_group)
                    for i in range(matrix.shape[0]):
                        if i == 6:
                            if now_layer == layers:
                                adj_matrix[i + node_start_no][self.nodes - 1] = 1  # to classifier
                            else:
                                adj_matrix[i + node_start_no][
                                    i + node_start_no + 1] = 1  # output node to next input node
                        else:
                            for j in range(matrix.shape[1]):
                                if matrix[i][j] == 1:
                                    adj_matrix[i + node_start_no][j + node_start_no] = 1


            # Edges E
            e = np.zeros((self.nodes, self.nodes, 1), dtype=float)
            spec = ModelSpec(np.array(matrix), ops)

            for now_layer in range(layers + 1):
                if now_layer == 0:
                    if now_layer == layers:
                        e[0][self.nodes - 1][0] = 128  # to classifier
                    else:
                        e[0][1][0] = 128  # stem to input node
                elif now_layer == 4:
                    e[21][22][0] = 128  # output to maxpool
                    if now_layer == layers:
                        e[22][self.nodes - 1][0] = 128
                    else:
                        e[22][23][0] = 128  # maxpool to input
                elif now_layer == 8:
                    e[43][44][0] = 256  # output to maxpool
                    if now_layer == layers:
                        e[44][self.nodes - 1][0] = 256
                    else:
                        e[44][45][0] = 256  # maxpool to input
                else:
                    now_group = now_layer // 4 + 1
                    node_start_no = now_group + 7 * (now_layer - now_group)
                    now_channel = now_group * 128

                    if now_layer == 1:
                        tmp_channels = compute_vertex_channels(now_channel, now_channel, spec.matrix)
                    elif now_layer == 5 or now_layer == 9:
                        tmp_channels = compute_vertex_channels(now_channel // 2, now_channel, spec.matrix)
                    else:
                        tmp_channels = compute_vertex_channels(now_channel, now_channel, spec.matrix)

                    # fix channels length to nodes num
                    node_channels = [0] * self.n_features
                    if len(node_channels) == len(tmp_channels):
                        node_channels = tmp_channels
                    else:
                        now_cot = 0
                        for n in range(len(tmp_channels)):
                            if np.all(matrix[now_cot] == 0) and now_cot != 6:
                                now_cot += 1
                                node_channels[now_cot] = tmp_channels[n]
                            else:
                                node_channels[now_cot] = tmp_channels[n]
                            now_cot += 1

                    for i in range(matrix.shape[0]):
                        if i == 6:  # output node to next input node
                            if now_layer == layers:
                                e[i + node_start_no][self.nodes - 1][0] = now_channel
                            else:
                                e[i + node_start_no][i + node_start_no + 1][0] = now_channel
                        else:
                            for j in range(matrix.shape[1]):
                                if matrix[i][j] == 1:
                                    e[i + node_start_no][j + node_start_no][0] = node_channels[j]

            graph_list.append(Graph(a=adj_matrix, e=e, x=x, y=y))

        return graph_list


if __name__ == '__main__':
    file = open('../incremental/cifar10_log/cifar10.pkl', 'rb')
    record = pickle.load(file)
    file.close()

    dataset = LearningCurveDataset(record_dic=record, record_dir='../incremental/cifar10_log/', num_samples=2000)
    dataset.read()
    print(dataset[0])
