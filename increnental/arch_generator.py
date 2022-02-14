import itertools


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
    if(end >= len(ret)):
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
    arch_list = generate_arch(amount_of_cell_layers=3, start=0, end=1)
    print(len(arch_list))
    print(arch_list)