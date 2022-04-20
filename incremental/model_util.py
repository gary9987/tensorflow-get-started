import pickle
import tensorflow as tf
import random
import numpy as np
from model_builder import build_arch_model
from model_spec import ModelSpec
from os import path
import os


def get_model_by_id_and_layer(cell_filename, shuffle_seed: int, inputs_shape: tuple, id: int, layer: int):

    # Auto download if cell_list.pkl is not exist
    if not path.exists(cell_filename):
        os.system('sh download.sh')
    file = open(cell_filename, 'rb')
    cell_list = pickle.load(file)
    file.close()
    random.seed(shuffle_seed)
    random.shuffle(cell_list)

    matrix, ops = cell_list[id][0], cell_list[id][1]

    spec = ModelSpec(np.array(matrix), ops)
    ori_model = build_arch_model(spec, inputs_shape)
    ori_model.build([*inputs_shape])

    model = tf.keras.Sequential()
    for layer_no in range(layer):
        model.add(ori_model.layers[layer_no])

    model.build([*inputs_shape])
    return model


def get_model_by_id(cell_filename, shuffle_seed: int, inputs_shape: tuple, id: int):

    # Auto download if cell_list.pkl is not exist
    if not path.exists(cell_filename):
        os.system('sh download.sh')
    file = open(cell_filename, 'rb')
    cell_list = pickle.load(file)
    file.close()
    random.seed(shuffle_seed)
    random.shuffle(cell_list)

    matrix, ops = cell_list[id][0], cell_list[id][1]

    spec = ModelSpec(np.array(matrix), ops)
    ori_model = build_arch_model(spec, inputs_shape)
    ori_model.build([*inputs_shape])

    return ori_model


def copy_model_weight(dst, src):
    assert len(dst.layers) <= len(src.layers)
    for i in range(len(dst.layers)):
        dst.layers[i].set_weights(src.layers[i].get_weights())
    return dst


if __name__ == '__main__':
    ori_model = get_model_by_id('./cell_list.pkl', shuffle_seed=0, inputs_shape=(None, 28, 28, 1), id=0)
    print(ori_model.summary())
    sub_model = get_model_by_id_and_layer('./cell_list.pkl', shuffle_seed=0, inputs_shape=(None, 28, 28, 1), id=0, layer=6)
    print(sub_model.summary())

    sub_model = copy_model_weight(sub_model, ori_model)

