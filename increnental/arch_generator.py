import itertools

def generate_cell(amount_of_layer, start, end):
    layer_type = ['Conv2D 64 3 3 same 1 1', 'Conv2D 64 1 1 same 1 1', 'MaxPooling2D 3 3 same 1 1']
    if start > end:
        return []
    ret = []
    for cell in itertools.product(layer_type, repeat=amount_of_layer):
        ret.append(cell)
    if(end >= len(ret)):
        print("The \"end\" is out of bound. Amount of layer:", amount_of_layer, " Total have", len(ret), "possibility.")
        return []

    return ret[start:end + 1]


if __name__ == '__main__':
    cell_list = generate_cell(7, 0, 2222)
    print(len(cell_list))
    print(cell_list)