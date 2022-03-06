import itertools
import csv
import hashlib
import numpy as np
import pickle
from model_builder import Cell_Model
import model_spec


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
    not_valid = 0
    for cell in itertools.product(*[x for x in rowx]):
        try:
            if is_valid_matrix(list(cell)):
                model_spec.ModelSpec(matrix=list(cell), ops=['INPUT', 'CONV1X1', 'CONV3X3', 'CONV3X3', 'CONV3X3', 'MAXPOOL3X3', 'OUTPUT'])
                matrix_list.append(list(cell))
        except:
            not_valid += 1

    print("Not valid: ", not_valid)
    return matrix_list


def matrix_to_arch_path(matrix, ops=None):
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


def dump_matrix_list(size=7, filename='./matrix_list.pkl'):
    matrix_list = generate_valid_matrix(size=size)
    with open(filename, 'wb') as f:
        pickle.dump(matrix_list, f)


def dump_cell_list(size=7, filename='./arch_list.pkl'):
    file = open('./matrix_list.pkl', 'rb')
    matrix_list = pickle.load(file)
    file.close()

    arch_count_map = {}
    record = []

    ops_type = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
    ops_type = [ops_type] * (size - 2)

    len_ops = 0
    for _ in itertools.product(['INPUT'], *ops_type, ['OUTPUT']):
        len_ops += 1

    print(len(matrix_list))
    print(len_ops)

    for ops in itertools.product(['INPUT'], *ops_type, ['OUTPUT']):
        print(ops)
        for matrix in matrix_list:

            ops = list(ops)
            # Get the matrix with ops dfs path to detect isomorphism
            arch_path = matrix_to_arch_path(matrix, ops)

            if len(arch_path) != 0:
                arch_path.sort()
                arch_hash = hashlib.shake_128(str(arch_path).encode('utf-8')).hexdigest(10)
                if arch_count_map.get(arch_hash) is None:
                    arch_count_map[arch_hash] = arch_path
                    record.append([matrix, ops])

    print(len(record))
    with open(filename, 'wb') as f:
        pickle.dump(record, f)

'''
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
'''

if __name__ == '__main__':
    #dump_matrix_list()
    #dump_cell_list(7)
    file = open('./arch_list.pkl', 'rb')
    cell_list = pickle.load(file)
    file.close()
    print(len(cell_list))
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
