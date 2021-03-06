import logging
import random
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
import os
import wget
from keras import backend as K
from os import path
import re
import hashlib
import model_builder

logging.basicConfig(filename='nas_bench_101_dataset.log', level=logging.INFO)


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


def get_layerlist(model):
    # print('test for dir(model)')
    # print(dir(model))

    layerlist = []
    for layer_no in range(len(model.layers)):

        # have sublayer
        if 'model' in model.layers[layer_no].name:
            temp = model.get_layer(name=model.layers[layer_no].name)

            # print(dir(temp))
            # print(model.layers[layer_no].get_config())
            # break

            # print(temp.ops)
            # print('in submodel {}'.format(model.layers[layer_no].name))
            for sublayer_no in range(1, len(temp.ops) + 1):
                layerlist.append(temp.ops[sublayer_no].name)
                # print(temp.ops[sublayer_no].name)
                # print(dir(temp.layers[sublayer_no]))
                # print(temp.ops[sublayer_no].get_config())
                # print(temp.layers[sublayer_no].from_config())
                # print(temp.ops[sublayer_no].count_params())

                # print('FLOPS: {}, Params: {}'.format(get_flops(model, temp.ops[sublayer_no].name), get_params(temp.ops[sublayer_no])))

                # print(dir(temp.ops[sublayer_no]))
                # print('count_params: {}, trainable_variables: {}'.format(temp.ops[sublayer_no].count_params(), temp.ops[sublayer_no].trainable_variables))
                # print('*'*40)
        # no sublayer
        else:
            layerlist.append(model.layers[layer_no].name)
            # print(model.layers[layer_no].name)
            # print(model.layers[layer_no].get_config)
            # print('FLOPS: {}, Params: {}'.format(get_flops(model, model.layers[layer_no].name), get_params(model.layers[layer_no])))

            # print(dir(model.layers[layer_no]))
            # print('count_params: {}, trainable_variables: {}'.format(model.layers[layer_no].count_params(), model.layers[layer_no].trainable_variables))
            # print('*'*40)
    return layerlist


def get_layerlist_compare(model):
    # print('test for dir(model)')
    # print(dir(model))

    layerlist = []
    for layer_no in range(len(model.layers)):

        # have sublayer
        if 'model' in model.layers[layer_no].name:
            temp = model.get_layer(name=model.layers[layer_no].name)

            # print(dir(temp))
            # print(model.layers[layer_no].get_config())
            # break

            # print(temp.ops)
            # print('in submodel {}'.format(model.layers[layer_no].name))
            for sublayer_no in range(len(temp.layers)):
                layerlist.append(temp.layers[sublayer_no].name)
                # print(temp.ops[sublayer_no].name)
                # print(dir(temp.layers[sublayer_no]))
                # print(temp.ops[sublayer_no].get_config())
                # print(temp.layers[sublayer_no].from_config())
                # print(temp.ops[sublayer_no].count_params())

                # print('FLOPS: {}, Params: {}'.format(get_flops(model, temp.ops[sublayer_no].name), get_params(temp.ops[sublayer_no])))

                # print(dir(temp.ops[sublayer_no]))
                # print('count_params: {}, trainable_variables: {}'.format(temp.ops[sublayer_no].count_params(), temp.ops[sublayer_no].trainable_variables))
                # print('*'*40)
        # no sublayer
        else:
            layerlist.append(model.layers[layer_no].name)
            # print(model.layers[layer_no].name)
            # print(model.layers[layer_no].get_config)
            # print('FLOPS: {}, Params: {}'.format(get_flops(model, model.layers[layer_no].name), get_params(model.layers[layer_no])))

            # print(dir(model.layers[layer_no]))
            # print('count_params: {}, trainable_variables: {}'.format(model.layers[layer_no].count_params(), model.layers[layer_no].trainable_variables))
            # print('*'*40)
    return layerlist


def to_number(str):
    if 'm' in str:
        temp = str.strip('m')
        return int(float(temp) * 1000000)

    elif 'k' in str:
        temp = str.strip('k')
        return int(float(temp) * 1000)
    elif 'g':
        temp = str.strip('g')
        return int(float(temp) * 1000000000)


def find_flops(contents, layername):
    target = '.*' + layername + '/.*'
    for line in contents:
        # print(line)

        if re.match(target, line):
            # print(line)
            temp = line.split('(')
            temp = temp[1].split(' ')
            temp = temp[0].split('/')
            out = temp[1]

            return to_number(out)


def get_flops(model, profile_name, layername):
    profile_filename = profile_name
    if path.exists(profile_filename):
        file = open(profile_filename, 'r')
        contents = file.readlines()
        ff = find_flops(contents, layername)
        return str(ff)

    layerlist = get_layerlist(model)
    # print('origin model layerlist')
    # print(layerlist)

    for i in range(0, len(layerlist)):
        layerlist[i] = '.*' + layerlist[i] + '/.*'

    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()

        '''
        opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
                tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()  )
                .with_node_names(show_name_regexes=layerlist)
                .with_stdout_output()
                .build())
        '''
        opts = (tf.compat.v1.profiler.ProfileOptionBuilder(
            tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
                .with_node_names(show_name_regexes=layerlist)
                .with_file_output(profile_filename)
                .build())
        # '''

        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="scope", options=opts)

        file = open(profile_filename, 'r')
        contents = file.readlines()
        ff = find_flops(contents, layername)
        return str(ff)

        # return flops.total_float_ops


def get_params(layer):
    return layer.count_params()


def get_tensor_shape(model, layername):
    # return [input_shape, output_shape]
    for layer_no in range(len(model.layers)):

        # have sublayer
        if 'model' in model.layers[layer_no].name:
            temp = model.get_layer(name=model.layers[layer_no].name)

            for sublayer_no in range(len(temp.layers)):
                # print(temp.layers[sublayer_no].name)
                if temp.layers[sublayer_no].name == layername:
                    # print(temp.layers[sublayer_no].input_shape)
                    return [temp.layers[sublayer_no].input_shape, temp.layers[sublayer_no].output_shape]

        # no sublayer
        else:
            # print(model.layers[layer_no].name)
            if model.layers[layer_no].name == layername:
                return [model.layers[layer_no].input_shape, model.layers[layer_no].output_shape]

    print('error')


def get_hash(id, layer):
    # calculate hash key of this model
    model_string = str(id) + '_' + str(layer)
    hash_sha256 = hashlib.sha256()
    hash_sha256.update(model_string.encode("utf-8"))
    hash = hash_sha256.hexdigest()

    return hash


def get_model_by_id_and_layer_original(cell_filename, shuffle_seed: int, inputs_shape: tuple, id: int, layer: int):
    '''
    # Auto download if cell_list.pkl is not exist
    if not path.exists(cell_filename):
        os.system('sh download.sh')
    '''
    file = open(cell_filename, 'rb')
    cell_list = pickle.load(file)
    file.close()
    random.seed(shuffle_seed)
    random.shuffle(cell_list)

    matrix, ops = cell_list[id][0], cell_list[id][1]

    spec = ModelSpec(np.array(matrix), ops)
    ori_model = model_builder.build_arch_model_original(spec, inputs_shape)
    ori_model.build([*inputs_shape])

    model = tf.keras.Sequential()
    # layer index is 0-based
    for layer_no in range(layer + 1):
        model.add(ori_model.layers[layer_no])

    # model.add(tf.keras.Sequential([Classifier(num_classes, spec.data_format)]))

    model.build([*inputs_shape])
    return model


class NasBench101Dataset(Dataset):
    def __init__(self, start, end, record_dic=None, shuffle_seed=0, inputs_shape=None, num_classes=10, **kwargs):
        """
        :param start: The start index of data you want to query.
        :param end: The end index of data you want to query.
        :param record_dic: open('./nas-bench-101-data/nasbench_101_cell_list.pkl', 'rb')
        :param shuffle_seed: 0
        :param inputs_shape: (None, 32, 32, 3)
        :param num_classes: Number of the classes of the dataset

        Direct use the dataset with set the start and end parameters,
        or if you want to preprocess again, unmark the marked download() function and set the all parameters.
        """
        self.nodes = 67
        self.features_dict = {'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4,
                              'Classifier': 5, 'maxpool2x2': 6, 'flops': 7, 'params': 8, 'num_layer': 9,
                              'input_shape_1': 10, 'input_shape_2': 11, 'input_shape_3': 12, 'output_shape_1': 13,
                              'output_shape_2': 14, 'output_shape_3': 15}

        self.num_features = len(self.features_dict)
        self.inputs_shape = inputs_shape
        self.num_classes = num_classes
        self.start = start
        self.end = end
        self.file_path = 'NasBench101Dataset'
        self.shuffle_seed = shuffle_seed
        self.cell_filename = './nas-bench-101-data/nasbench_101_cell_list.pkl'
        self.total_layers = 11
        self.record_dic = record_dic

        if self.record_dic is not None:
            random.seed(shuffle_seed)
            random.shuffle(self.record_dic)

        super().__init__(**kwargs)

    def download(self):
        if not os.path.exists(self.file_path):
            print('Downloading...')
            file_name = wget.download('https://www.dropbox.com/s/40lrvb3lcgij5c8/NasBench101Dataset.zip?dl=1')
            print('Save dataset to {}'.format(file_name))
            os.system('unzip {}'.format(file_name))
            print(f'Unzip dataset finish.')

    '''
    def download(self):  # preprocessing
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)

        profile_dir = './tmp_profile/'
        if not os.path.exists(profile_dir):
            os.mkdir(profile_dir)

        total_layers = self.total_layers

        for record, no in zip(self.record_dic[self.start: self.end + 1], range(self.start, self.end + 1)):
            if os.path.exists(os.path.join(self.file_path, f'graph_{no}.npz')):
                continue

            matrix, ops, layers = np.array(record[0]), record[1], total_layers
            spec = ModelSpec(np.array(matrix), ops)
            num_nodes = matrix.shape[0]
            # get model
            K.clear_session()
            model = get_model_by_id_and_layer_original(self.cell_filename, shuffle_seed=self.shuffle_seed,
                                                       inputs_shape=self.inputs_shape, id=no, layer=total_layers)
            model.add(tf.keras.Sequential([Classifier(self.num_classes, spec.data_format)]))

            # get profile_filename for speed up get_flops
            profile_name = profile_dir + get_hash(id=no, layer=total_layers) + '.txt'
            # restart layer counter
            K.clear_session()
            # get model_tensor
            # used for get_tensor_shape
            model_tensor = model_util.get_model_by_id_and_layer(self.cell_filename, shuffle_seed=self.shuffle_seed,
                                                                inputs_shape=self.inputs_shape, id=no, layer=total_layers)
            model_tensor.add(tf.keras.Sequential([Classifier(self.num_classes, spec.data_format)]))

            # Labels Y
            y = np.zeros((3, 4))  # 3 x (train_accuracy, validation_accuracy, test_accuracy, training_time)
            for query_idx in range(3):
                y[query_idx][0] = record[2][query_idx]['train_accuracy']
                y[query_idx][1] = record[2][query_idx]['validation_accuracy']
                y[query_idx][2] = record[2][query_idx]['test_accuracy']
                y[query_idx][3] = record[2][query_idx]['training_time']

            # Node features X
            x = np.zeros((self.nodes, self.num_features), dtype=float)  # nodes * (features + metadata + num_layer)

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

            accumulation_layer = 0
            for now_layer in range(11 + 1):
                if now_layer == 0 or now_layer == 4 or now_layer == 8:
                    if now_layer == 0:
                        offset_idx = 0
                    elif now_layer == 4:
                        offset_idx = 22
                    else:  # now_layer == 8
                        offset_idx = 44

                    accumulation_layer += 1
                    x[offset_idx][self.features_dict['num_layer']] = accumulation_layer
                    x[offset_idx][self.features_dict['conv3x3-bn-relu']] = 1  # stem is a 'conv3x3-bn-relu' type
                    x[offset_idx][self.features_dict['flops']] = get_flops(model, profile_name, model.layers[now_layer].name)
                    x[offset_idx][self.features_dict['params']] = get_params(model.layers[now_layer])
                    tensor = get_tensor_shape(model_tensor, model.layers[now_layer].name)
                    input_shape, output_shape = tensor[0], tensor[1]
                    for dim in range(1, 3+1):
                        x[offset_idx][self.features_dict['input_shape_{}'.format(str(dim))]] = input_shape[dim]
                        x[offset_idx][self.features_dict['output_shape_{}'.format(str(dim))]] = output_shape[dim]
                else:
                    cell_layer = model.get_layer(name=model.layers[now_layer].name)
                    now_group = now_layer // 4 + 1
                    node_start_no = now_group + 7 * (now_layer - now_group)

                    skip_cot = 0
                    for i in range(len(ops)):
                        x[i + node_start_no][self.features_dict['num_layer']] = accumulation_layer + node_depth[i]
                        x[i + node_start_no][self.features_dict[ops[i]]] = 1
                        if 1 <= i <= len(cell_layer.ops):  # cell_layer ops
                            if len(cell_layer.ops) == 5:  # no need to skip
                                offset_idx = i + node_start_no
                            else:
                                if np.all(matrix[i] == 0):
                                    skip_cot += 1
                                offset_idx = i + node_start_no + skip_cot

                            x[offset_idx][self.features_dict['flops']] = \
                                get_flops(model, profile_name, cell_layer.ops[i].name)
                            x[offset_idx][self.features_dict['params']] = get_params(cell_layer.ops[i])
                            tensor = get_tensor_shape(model_tensor, cell_layer.ops[i].name)
                            input_shape, output_shape = tensor[0], tensor[1]
                            for dim in range(1, 3 + 1):
                                x[offset_idx][self.features_dict['input_shape_{}'.format(str(dim))]] = input_shape[dim]
                                x[offset_idx][self.features_dict['output_shape_{}'.format(str(dim))]] = output_shape[dim]

                    accumulation_layer += node_depth[-1]

            x[66][self.features_dict['num_layer']] = accumulation_layer + 1
            x[66][self.features_dict['Classifier']] = 1
            x[66][self.features_dict['flops']] = get_flops(model, profile_name, model.layers[12].name)
            x[66][self.features_dict['params']] = get_params(model.layers[12])
            tensor = get_tensor_shape(model_tensor, model.layers[12].name)
            input_shape, output_shape = tensor[0], tensor[1]
            for dim in range(1, 3 + 1):
                x[66][self.features_dict['input_shape_{}'.format(str(dim))]] = input_shape[dim]

            # The output_shape of classifier is only two dimension ([None, classes_num])
            x[66][self.features_dict['output_shape_1']] = output_shape[1]

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
            logging.info(f'graph_{no}.npz is saved.')
    '''

    def read(self):
        output = []
        for i in range(self.start, self.end + 1):
            data = np.load(os.path.join(self.file_path, f'graph_{i}.npz'))
            output.append(
                Graph(x=data['x'], e=data['e'], a=data['a'], y=data['y'])
            )
        return output


if __name__ == '__main__':
    file = open('./nas-bench-101-data/nasbench_101_cell_list.pkl', 'rb')
    record = pickle.load(file)
    file.close()

    #dataset = NasBench101Dataset(record_dic=record, shuffle_seed=0, start=0,
    #                             end=len(record), inputs_shape=(None, 32, 32, 3), num_classes=10)

    # Test read()
    dataset = NasBench101Dataset(record_dic=record, shuffle_seed=0, start=0,
                                 end=5000, inputs_shape=(None, 32, 32, 3), num_classes=10)
