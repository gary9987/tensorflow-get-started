import itertools
import csv
import hashlib
import os.path
from pathlib import Path

import numpy as np
import pickle
from model_builder import CellModel
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


class Matrix_Generator:
    def __init__(self, size, edge_limit, data_dir='cell_list_data'):
        self.size = size
        self.edge_limit = edge_limit
        self.matrix_list = []
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir()

        self.map_size2ops = {2: ['INPUT', 'OUTPUT'],
                        3: ['INPUT', 'CONV1X1', 'OUTPUT'],
                        4: ['INPUT', 'CONV1X1', 'CONV3X3', 'OUTPUT'],
                        5: ['INPUT', 'CONV1X1', 'CONV3X3', 'CONV3X3', 'OUTPUT'],
                        6: ['INPUT', 'CONV1X1', 'CONV3X3', 'CONV3X3', 'CONV3X3', 'OUTPUT'],
                        7: ['INPUT', 'CONV1X1', 'CONV3X3', 'CONV3X3', 'CONV3X3', 'MAXPOOL3X3', 'OUTPUT']}

    def generate_valid_matrix(self):
        rowx = [list() for _ in range(self.size)]
        for i in range(self.size, 0, -1):
            for j in range(2 ** (i - 1)):
                bin = "{j:0{size}b}".format(size=self.size, j=j)
                bin_to_list = [int(x) for x in bin]
                rowx[self.size - i].append(bin_to_list)

        # Remove the all 0 row, because this means input vertex not connect to any vertex
        rowx[0].remove([0 for _ in range(self.size)])

        not_valid = 0
        for cell in itertools.product(*[x for x in rowx]):
            try:
                if sum(sum(list(cell), [])) <= self.edge_limit and is_valid_matrix(list(cell)):
                    model_spec.ModelSpec(matrix=list(cell), ops=self.map_size2ops[self.size])
                    self.matrix_list.append(list(cell))
            except:
                not_valid += 1

        print('Matrix list length is', len(self.matrix_list))
        print("Invalid amount: ", not_valid)

    def dump_to_file(self, filename='./matrix_list.pkl'):
        if len(self.matrix_list) == 0:
            print('Please call generate_valid_matrix() first')
            return

        file_path = self.data_dir / Path(filename)
        with open(file_path, 'wb') as f:
            pickle.dump(self.matrix_list, f)

        print(str(file_path))
        return str(file_path)


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


def dump_cell_list_by_matrix_list(size=7, in_matrix_filename='./matrix_list.pkl', out_cell_list_filename='./cell_list.pkl'):
    file = open(in_matrix_filename, 'rb')
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

    print('Cell list length is', len(record))
    with open(out_cell_list_filename, 'wb') as f:
        pickle.dump(record, f)
    print(out_cell_list_filename)


if __name__ == '__main__':
    data_dir = 'cell_list_data'
    size = 7
    matrix_filename = f'matrix_list_{size}.pkl'
    cell_list_filename = f'cell_list_{size}.pkl'

    matrix_generator = Matrix_Generator(size=size, edge_limit=9, data_dir=data_dir)
    matrix_generator.generate_valid_matrix()
    matrix_generator.dump_to_file(filename=matrix_filename)

    dump_cell_list_by_matrix_list(size, os.path.join(data_dir, matrix_filename), os.path.join(data_dir, cell_list_filename))

    file = open(os.path.join(data_dir, cell_list_filename), 'rb')
    cell_list = pickle.load(file)
    file.close()
    print(cell_list[0])

