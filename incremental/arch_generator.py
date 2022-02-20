import itertools
import csv
import hashlib

def generate_matrix():
    matrix_list = []

    rowx = [list() for _ in range(7)]
    for i in range(7, 0, -1):
        for j in range(2 ** (i - 1)):
            bin = "{:07b}".format(j)
            bin_to_list = [int(x) for x in bin]
            rowx[7 - i].append(bin_to_list)

    rowx[0].remove([0 for _ in range(7)])
    for cell in itertools.product(rowx[0], rowx[1], rowx[2], rowx[3], rowx[4], rowx[5], rowx[6]):
        matrix_list.append(list(cell))

    return matrix_list


def matrix_to_arch(matrix, ops=None):
    if ops is None:
        ops = ['INPUT', 'CONV1X1', 'CONV3X3', 'CONV3X3', 'CONV3X3', 'MAXPOOL3X3', 'OUTPUT']

    arch = []

    def build_with_dfs(abranch: list, ind):
        if ops[ind] == 'OUTPUT' and len(abranch) != 0:
            arch.append(abranch.copy())
            return

        abranch.append(ops[ind])
        for k in range(len(matrix[ind])):
            if matrix[ind][k] == 1:
                build_with_dfs(abranch, k)
        abranch.pop()

    for i in range(len(matrix[0])):
        if matrix[0][i] == 1:
            tmp = []
            build_with_dfs(tmp, i)

    return arch


def dump_arch_list(filename='./arch_list.csv'):
    matrix_list = generate_matrix()

    arch_count_map = {}
    for matrix in matrix_list:
        arch = matrix_to_arch(matrix)
        if len(arch) != 0:
            arch.sort()
            arch_hash = hashlib.shake_128(str(arch).encode('utf-8')).hexdigest(10)
            if arch_count_map.get(arch_hash) is None:
                arch_count_map[arch_hash] = arch

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        for k, v in arch_count_map.items():
            writer.writerow([v])


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
    dump_arch_list()
