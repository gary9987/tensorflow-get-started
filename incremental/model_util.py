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

    return model






