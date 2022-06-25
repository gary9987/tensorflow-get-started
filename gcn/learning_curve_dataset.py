from spektral.data import Dataset, Graph
import pickle
import numpy as np
import csv
from model_spec import ModelSpec
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import model_util
from classifier import Classifier
from model_builder import build_arch_model
import os
import wget


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


def get_flops(model, layername):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()

        opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
                .with_node_names(show_name_regexes=['.*' + layername + '/.*'])
                .build())
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="scope", options=opts)

        return flops.total_float_ops


def get_params(layer):
    return layer.count_params()


class LearningCurveDataset(Dataset):
    def __init__(self, record_dic, record_dir, inputs_shape, num_classes, start, end, **kwargs):
        self.nodes = 67
        self.num_features = 9
        self.inputs_shape = inputs_shape
        self.num_classes = num_classes
        self.start = start
        self.end = end
        self.file_path = 'LearningCurveDataset'
        # 'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4, 'Classifier': 5,
        # 'maxpool2x2': 6,
        self.features_dict = {'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4,
                              'Classifier': 5, 'maxpool2x2': 6, 'flops': 7, 'params': 8, 'num_layer': 9}
        self.record_dir = record_dir
        self.record_dic = record_dic
        super().__init__(**kwargs)

    def download(self):
        if not os.path.exists(self.file_path):
            print('Downloading...')
            file_name = wget.download('https://www.dropbox.com/s/8crtrh8weuoi5d2/LearningCurveDataset.zip?dl=1')
            os.system('unzip {}'.format(file_name))
            print('Save dataset to {}'.format(file_name))

    def read(self):
        output = []
        for i in range(self.start, self.end + 1):
            data = np.load(os.path.join(self.file_path, f'graph_{i}.npz'))
            output.append(
                Graph(x=data['x'], e=data['e'], a=data['a'], y=data['y'])
            )
        return output

    '''
    def download(self):  # preprocessing
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

        for record, no in zip(self.record_dic[self.start: self.end + 1], range(self.start, self.end + 1)):
            if os.path.exists(os.path.join(self.file_path, f'graph_{no}.npz')):
                continue

            matrix, ops, layers, log_file = np.array(record['matrix']), record['ops'], record['layers'], record[
                'log_file']
            spec = ModelSpec(np.array(matrix), ops)
            num_nodes = matrix.shape[0]
            # build model for get metadata in part of Node features
            model = build_arch_model(spec, self.inputs_shape)
            model.add(tf.keras.Sequential([Classifier(self.num_classes, spec.data_format)]))
            model.build([*self.inputs_shape])

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
            x = np.zeros((self.nodes, self.num_features), dtype=float)  # nodes * (features + metadata)
            for now_layer in range(11 + 1):
                if now_layer == 0:
                    x[0][self.features_dict['conv3x3-bn-relu']] = 1  # stem is a 'conv3x3-bn-relu' type
                    x[0][self.features_dict['flops']] = get_flops(model, model.layers[now_layer].name)
                    x[0][self.features_dict['params']] = get_params(model.layers[now_layer])
                elif now_layer == 4:
                    x[22][self.features_dict['maxpool2x2']] = 1  # maxpool2x2
                    x[22][self.features_dict['flops']] = get_flops(model, model.layers[now_layer].name)
                    x[22][self.features_dict['params']] = get_params(model.layers[now_layer])
                elif now_layer == 8:
                    x[44][self.features_dict['maxpool2x2']] = 1  # maxpool2x2
                    x[44][self.features_dict['flops']] = get_flops(model, model.layers[now_layer].name)
                    x[44][self.features_dict['params']] = get_params(model.layers[now_layer])
                else:
                    cell_layer = model.get_layer(name=model.layers[now_layer].name)
                    now_group = now_layer // 4 + 1
                    node_start_no = now_group + 7 * (now_layer - now_group)

                    skip_cot = 0
                    for i in range(len(ops)):
                        x[i + node_start_no][self.features_dict[ops[i]]] = 1
                        if 1 <= i <= len(cell_layer.ops):  # cell_layer ops
                            if len(cell_layer.ops) == 5:  # no need to skip
                                x[i + node_start_no][self.features_dict['flops']] = get_flops(model, cell_layer.ops[i].name)
                                x[i + node_start_no][self.features_dict['params']] = get_params(cell_layer.ops[i])
                            else:
                                if np.all(matrix[i] == 0):
                                    skip_cot += 1
                                x[i + node_start_no + skip_cot][self.features_dict['flops']] = \
                                    get_flops(model, cell_layer.ops[i].name)
                                x[i + node_start_no + skip_cot][self.features_dict['params']] = get_params(cell_layer.ops[i])

            x[66][self.features_dict['Classifier']] = 1
            x[66][self.features_dict['flops']] = get_flops(model, model.layers[12].name)
            x[66][self.features_dict['params']] = get_params(model.layers[12])

            # Adjacency matrix A
            adj_matrix = np.zeros((self.nodes, self.nodes), dtype=float)

            # 0 convbn 128
            # 1 cell
            # 8 cell
            # 15 cell
            # 22 maxpool
            # 23 cell
            # 30 cell
            # 37 cell
            # 44 maxpool
            # 45 cell
            # 52 cell
            # 59 cell

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

                    # fix channels length to number of nudes
                    node_channels = [0] * num_nodes
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

            filename = os.path.join(self.file_path, f'graph_{no}.npz')
            np.savez(filename, a=adj_matrix, x=x, e=e, y=y)
    '''


    '''
    def read(self):
        graph_list = []

        for record in self.record_dic[self.start: self.end + 1]:
            matrix, ops, layers, log_file = np.array(record['matrix']), record['ops'], record['layers'], record[
                'log_file']
            spec = ModelSpec(np.array(matrix), ops)
            num_nodes = matrix.shape[0]
            # build model for get metadata in part of Node features
            model = build_arch_model(spec, self.inputs_shape)
            model.add(tf.keras.Sequential([Classifier(self.num_classes, spec.data_format)]))
            model.build([*self.inputs_shape])

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
            x = np.zeros((self.nodes, self.num_features), dtype=float)  # nodes * (features + metadata)
            for now_layer in range(11 + 1):
                if now_layer == 0:
                    x[0][self.features_dict['conv3x3-bn-relu']] = 1  # stem is a 'conv3x3-bn-relu' type
                    x[0][self.features_dict['flops']] = get_flops(model, model.layers[now_layer].name)
                    x[0][self.features_dict['params']] = get_params(model.layers[now_layer])
                elif now_layer == 4:
                    x[22][self.features_dict['maxpool2x2']] = 1  # maxpool2x2
                    x[22][self.features_dict['flops']] = get_flops(model, model.layers[now_layer].name)
                    x[22][self.features_dict['params']] = get_params(model.layers[now_layer])
                elif now_layer == 8:
                    x[44][self.features_dict['maxpool2x2']] = 1  # maxpool2x2
                    x[44][self.features_dict['flops']] = get_flops(model, model.layers[now_layer].name)
                    x[44][self.features_dict['params']] = get_params(model.layers[now_layer])
                else:
                    cell_layer = model.get_layer(name=model.layers[now_layer].name)
                    now_group = now_layer // 4 + 1
                    node_start_no = now_group + 7 * (now_layer - now_group)
                    for i in range(len(ops)):
                        x[i + node_start_no][self.features_dict[ops[i]]] = 1
                        if 1 <= i <= 5:  # cell_layer ops
                            x[i + node_start_no][self.features_dict['flops']] = get_flops(model, cell_layer.ops[i].name)
                            x[i + node_start_no][self.features_dict['params']] = get_params(cell_layer.ops[i])

            x[66][self.features_dict['Classifier']] = 1
            x[66][self.features_dict['flops']] = get_flops(model, model.layers[12].name)
            x[66][self.features_dict['params']] = get_params(model.layers[12])

            # Adjacency matrix A
            adj_matrix = np.zeros((self.nodes, self.nodes), dtype=float)

            # 0 convbn 128
            # 1 cell
            # 8 cell
            # 15 cell
            # 22 maxpool
            # 23 cell
            # 30 cell
            # 37 cell
            # 44 maxpool
            # 45 cell
            # 52 cell
            # 59 cell

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

                    # fix channels length to number of nudes
                    node_channels = [0] * num_nodes
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
        '''

    def add_num_layer_to_node_feature(self):  # preprocessing

        for record, no in zip(self.record_dic[self.start: self.end + 1], range(self.start, self.end + 1)):
            if not os.path.exists(os.path.join(self.file_path, f'graph_{no}.npz')):
                print('Error, graph_{no}.npz not exit.')
                exit()

            matrix, ops, layers, log_file = np.array(record['matrix']), record['ops'], record['layers'], record[
                'log_file']

            data = np.load(os.path.join(self.file_path, f'graph_{no}.npz'))
            x = data['x']
            e = data['e']
            a = data['a']
            y = data['y']

            # already be processed
            if x.shape[1] > 9:
                continue

            num_nodes = matrix.shape[0]
            node_depth = [0] * num_nodes
            node_depth[0] = 1

            def dfs(node: int, now_depth: int):
                now_depth += 1
                for node_idx in range(num_nodes):
                    if matrix[node][node_idx] != 0:
                        node_depth[node_idx] = max(node_depth[node_idx], now_depth)
                        dfs(node_idx, now_depth)

            # Calculate the depth of each node
            dfs(0, 1)
            # [0, 1, 2, 1, 2, 3, 4]

            # num layer indicate that the node is on the layer i (1-base).

            # Add an additional column for x
            tmp_x = np.zeros((x.shape[0], x.shape[1] + 1))
            tmp_x[:, :-1] = x
            x = tmp_x

            # Node features X
            accumulation_layer = 0
            for now_layer in range(11 + 1):
                if now_layer == 0:
                    accumulation_layer += 1
                    x[0][self.features_dict['num_layer']] = accumulation_layer
                elif now_layer == 4:
                    accumulation_layer += 1
                    x[22][self.features_dict['num_layer']] = accumulation_layer
                elif now_layer == 8:
                    accumulation_layer += 1
                    x[44][self.features_dict['num_layer']] = accumulation_layer
                else:
                    now_group = now_layer // 4 + 1
                    node_start_no = now_group + 7 * (now_layer - now_group)

                    for i in range(len(ops)):
                        x[i + node_start_no][self.features_dict['num_layer']] = accumulation_layer + node_depth[i]

                    accumulation_layer += node_depth[-1]

            x[66][self.features_dict['num_layer']] = accumulation_layer + 1

            filename = os.path.join(self.file_path, f'graph_{no}.npz')
            np.savez(filename, a=a, x=x, e=e, y=y)


if __name__ == '__main__':
    file = open('../incremental/cifar10_log/cifar10.pkl', 'rb')
    record = pickle.load(file)
    file.close()

    dataset = LearningCurveDataset(record_dic=record, record_dir='../incremental/cifar10_log/', start=0,
                                         end=29999, inputs_shape=(None, 32, 32, 3), num_classes=10)

    dataset.add_num_layer_to_node_feature()


