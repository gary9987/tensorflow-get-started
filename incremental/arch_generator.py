import itertools
import csv
import hashlib
import numpy as np
import pickle


def compute_vertex_channels(input_channels, output_channels, matrix):
    """Computes the number of channels at every vertex.
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


def is_valid_matrix(matrix):
    arch = []
    visit = [0] * len(matrix)
    visit[0] = 1
    out_valid = True

    def build_with_dfs(a_branch: list, ind):
        visit[ind] = 1
        if ind == len(matrix) - 1 and len(a_branch) != 0:
            arch.append(a_branch.copy())
            return True

        valid = True
        a_branch.append(ind)
        out_degree = False
        for k in range(len(matrix[ind])):
            if matrix[ind][k] == 1:
                # if this node point to other node then out_degree is True
                out_degree = True
                # check all out branch is valid
                valid = valid and build_with_dfs(a_branch, k)
        a_branch.pop()
        return False if not out_degree else valid

    def check_all_zero(ind):
        for i in matrix[ind]:
            if i == 1:
                return False
        return True

    for i in range(len(matrix[0])):
        if matrix[0][i] == 1:
            tmp = []
            out_valid = out_valid and build_with_dfs(tmp, i)

    # check matrix row of node which in_degree is 0 is all 0
    for i in range(len(visit)):
        if visit[i] == 0:
            out_valid = out_valid and check_all_zero(i)

    return out_valid


def generate_valid_matrix(size=7):
    matrix_list = []

    rowx = [list() for _ in range(size)]
    for i in range(size, 0, -1):
        for j in range(2 ** (i - 1)):
            bin = "{j:0{size}b}".format(size=size, j=j)
            bin_to_list = [int(x) for x in bin]
            rowx[size - i].append(bin_to_list)

    rowx[0].remove([0 for _ in range(size)])

    for cell in itertools.product(*[x for x in rowx]):
        if is_valid_matrix(list(cell)):
            matrix_list.append(list(cell))

    return matrix_list


def matrix_to_arch(matrix, ops=None):
    if ops is None:
        ops = ['INPUT', 'CONV1X1', 'CONV3X3', 'CONV3X3', 'CONV3X3', 'MAXPOOL3X3', 'OUTPUT']

    arch = []

    def build_with_dfs(a_branch: list, ind):
        if ops[ind] == 'OUTPUT' and len(a_branch) != 0:
            arch.append(a_branch.copy())
            return

        a_branch.append(ops[ind])
        for k in range(len(matrix[ind])):
            if matrix[ind][k] == 1:
                build_with_dfs(a_branch, k)
        a_branch.pop()

    for i in range(len(matrix[0])):
        if matrix[0][i] == 1:
            tmp = []
            build_with_dfs(tmp, i)

    return arch


def dump_matrix_list(filename='./matrix_list.pkl'):
    matrix_list = generate_valid_matrix()
    with open(filename, 'wb') as f:
        pickle.dump(matrix_list, f)


def dump_arch_list(size=7, filename='./arch_list.pkl'):
    file = open('./matrix_list.pkl', 'rb')
    matrix_list = pickle.load(file)
    file.close()

    arch_count_map = {}
    record = []

    ops_type = ['CONV1X1', 'CONV3X3', 'MAXPOOL3X3']
    ops_type = [ops_type] * (size - 2)
    for ops in itertools.product(['INPUT'], *ops_type, ['OUTPUT']):
        for matrix in matrix_list:
            try:
                arch = matrix_to_arch(matrix, ops)
                if len(arch) != 0:
                    arch.sort()
                    arch_hash = hashlib.shake_128(str(arch).encode('utf-8')).hexdigest(10)
                    if arch_count_map.get(arch_hash) is None:
                        arch_count_map[arch_hash] = arch
                        record.append([arch, matrix])
                    else:
                        print(arch)
            except:
                print('Not valid matrix')

    with open(filename, 'wb') as f:
        pickle.dump(record, f)



def generate_cell(amount_of_layer, start, end):
    """
    :param amount_of_layer: Means the amount of the layer in the cell.
    :param start: The start id of the cell you want to request.
    :param end: The end id of the cell you want to request.
    :return: list of cell include the id from start to end.
    """
    # Set the total layer types here.
    layer_type = ['Conv2D 64 3 3 same 1 1', 'Conv2D 64 1 1 same 1 1', 'MaxPooling2D 3 3 same 1 1']

    if start > end:
        return []
    ret = []
    for cell in itertools.product(layer_type, repeat=amount_of_layer):
        ret.append(list(cell))
    print("[Notice] Amount of layer", amount_of_layer, "has", len(ret), "possibility")
    if end >= len(ret):
        print("The \"end\" is out of bound. Amount of layer:", amount_of_layer, " Total have", len(ret), "possibility.")
        return []

    return ret[start:end + 1]


def generate_arch(amount_of_cell_layers, start, end):
    """
    :param amount_of_cell_layers: Means the amount of the layer fo each cell.
    :param start: The start id of the architecture you want to request.
    :param end: The end id of the architecture you want to request.
    :return: list of architecture include the id from start to end.
    """
    cell_list = generate_cell(amount_of_cell_layers, start, end)
    if cell_list == []:
        return cell_list

    arch_list = []
    for cell in cell_list:
        # The initial layer
        tmp_arch = ['Conv2D 128 3 3 same 1 1']
        tmp_arch += cell
        # Downsample layer
        tmp_arch.append('MaxPooling2D 2 2 valid 2 2')
        tmp_arch += cell
        # Downsample layer
        tmp_arch.append('MaxPooling2D 2 2 valid 2 2')
        tmp_arch += cell
        tmp_arch.append('GlobalAveragePooling2D')

        arch_list.append(tmp_arch)

    return arch_list


if __name__ == '__main__':
    dump_arch_list(7)
    # dump_matrix_list()


    '''
    matrix = [[0, 1, 1, 1, 0, 1, 0],  # input layer
              [0, 0, 0, 0, 0, 0, 1],  # 1x1 conv
              [0, 0, 0, 0, 0, 0, 1],  # 3x3 conv
              [0, 0, 0, 0, 1, 0, 0],  # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],  # 5x5 conv (replaced by two 3x3's)
              [0, 0, 0, 0, 0, 0, 1],  # 3x3 max-pool
              [0, 0, 0, 0, 0, 0, 0]]
    x = compute_vertex_channels(64, 128, np.array(matrix))
    print(x)
    '''
