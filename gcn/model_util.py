import pickle
import time
import timeit
from typing import Tuple, Union, Optional
from scipy.spatial.distance import hamming
import tensorflow as tf
import random
import numpy as np
from model_builder import build_arch_model, build_arch_model_original
from model_spec import ModelSpec
from os import path
import os


def get_model_by_spec(spec: ModelSpec, inputs_shape: tuple):
    model = build_arch_model_original(spec, inputs_shape)
    model.build([*inputs_shape])
    return model


def get_model_by_spec_and_layer(spec: ModelSpec, layer: int, inputs_shape: tuple):
    model = build_arch_model_original(spec, inputs_shape)
    model = tf.keras.models.Sequential(model.layers[:layer + 1])
    model.build([*inputs_shape])
    return model


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
    # layer index is 0-based
    for layer_no in range(layer+1):
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


def predict_to_bin_str_batch(model: tf.keras.Model, inputs: tf.Tensor) -> list:
    # Clone a new model and set weights by original model weights
    # private_model = tf.keras.models.clone_model(model)
    # private_model.set_weights(model.get_weights())
    # The function of added layer is equal to f(x) = (1 if x > 0 else 0)
    # private_model.add(tf.keras.layers.Lambda(lambda x: tf.sign(tf.maximum(x, 0))))
    # model_outs = list(tf.reshape(private_model.predict(inputs), (inputs.shape[0], -1)).numpy().astype(int))
    pred = model.predict(inputs)
    if model.layers[-1]._name.startswith('cell_model'):
        model_outs = model.layers[-1].intermediate_out.numpy().astype(int)
    else:
        pred = tf.keras.layers.Lambda(lambda x: tf.sign(tf.maximum(x, 0)))(pred)
        model_outs = np.reshape(pred, (inputs.shape[0], -1)).astype(int)

    return [''.join(str(e) for e in out) for out in model_outs]


def predict_to_bin_ndarray(model: tf.keras.Model, inputs: tf.Tensor) -> np.ndarray:
    # Clone a new model and set weights by original model weights
    # private_model = tf.keras.models.clone_model(model)
    # private_model.set_weights(model.get_weights())
    # The function of added layer is equal to f(x) = (1 if x > 0 else 0)
    # private_model.add(tf.keras.layers.Lambda(lambda x: tf.sign(tf.maximum(x, 0))))
    # model_outs = list(tf.reshape(private_model.predict(inputs), (inputs.shape[0], -1)).numpy().astype(int))
    pred = model.predict(inputs)
    if model.layers[-1]._name.startswith('cell_model'):
        model_outs = model.layers[-1].intermediate_out.numpy().astype(int)
    else:
        pred = tf.keras.layers.Lambda(lambda x: tf.sign(tf.maximum(x, 0)))(pred)
        model_outs = np.reshape(pred, (inputs.shape[0], -1)).astype(int)

    return model_outs


def hamming_distance(arr1: np.ndarray, arr2: np.ndarray) -> Tuple[float, int]:
    dis = hamming(arr1, arr2)
    return dis, 1


# A dataset 的難易程度 Model 5 layers
# 一個 data 各 layers 的 BinStr 全部接起來
# D1 = Data1 0|1|2|3|4|5
# D2 = Data2 0|1|2|3|4|5
# ... Dn
# 挑固定數量
# 計算分數
# row sample_size 兩兩做
# 投影片中的 N_A 是 H的上限
# Matrix = [[H(D1,D1), H(D1, D2)],
#           [[H(D2,D1), H(D2, D2)]]
# 之後算 det(Matrix)
'''
def calculate_dataset_level(cell_filename: str, shuffle_seed: int, inputs_shape: tuple, model_id: int,
                            data_samples: tf.Tensor):
    """
    This function is deprecated.
    """

    binstr_list = [""] * data_samples.shape[0]

    ori_model = get_model_by_id(cell_filename, shuffle_seed, inputs_shape, model_id)

    for i in range(len(ori_model.layers)):
        sub_model = get_model_by_id_and_layer(cell_filename, shuffle_seed, inputs_shape, model_id, layer=i)
        sub_model = copy_model_weight(dst=sub_model, src=ori_model)
        pred_list = predict_to_bin_str_batch(model=sub_model, inputs=data_samples)
        for j in range(len(pred_list)):
            binstr_list[j] += pred_list[j]

    matrix = np.zeros((data_samples.shape[0], data_samples.shape[0]))

    for i in range(data_samples.shape[0]):
        for j in range(data_samples.shape[0]):
            dis, maxn = hamming_distance(binstr_list[i], binstr_list[j])
            matrix[i][j] = maxn - dis

    det = np.linalg.det(matrix)

    return det
'''


def calculate_dataset_level(spec: ModelSpec, inputs_shape: tuple, data_samples: Union[tf.Tensor, np.ndarray]):
    bin_list = [np.array([])] * data_samples.shape[0]
    ori_model = get_model_by_spec(spec, inputs_shape)

    for i in range(len(ori_model.layers)):
        sub_model = get_model_by_spec_and_layer(spec, i, inputs_shape)
        sub_model = copy_model_weight(dst=sub_model, src=ori_model)
        pred_list = predict_to_bin_ndarray(model=sub_model, inputs=data_samples)

        for j in range(len(pred_list)):
            bin_list[j] = np.concatenate((bin_list[j], pred_list[j]), axis=0)

    bin_list = np.array(bin_list)

    def get_matrix_value(i, j):
        dis, maxn = hamming_distance(bin_list[i], bin_list[j])
        return maxn - dis

    matrix = [[get_matrix_value(i, j) for j in range(data_samples.shape[0])] for i in range(data_samples.shape[0])]
    matrix2 = np.array(matrix)
    det = np.linalg.det(matrix)
    return det


if __name__ == '__main__':
    '''
    ori_model = get_model_by_id('./cell_list.pkl', shuffle_seed=0, inputs_shape=(None, 28, 28, 1), id=0)
    print(ori_model.summary())
    sub_model = get_model_by_id_and_layer('./cell_list.pkl', shuffle_seed=0, inputs_shape=(None, 28, 28, 1), id=0,
                                          layer=1)
    print(sub_model.summary())
    sub_model = copy_model_weight(sub_model, ori_model)
    '''
    np.random.seed(0)
    input_shape = (5, 28, 28, 1)
    data = tf.convert_to_tensor(np.random.randint(128, size=input_shape))

    with open('nas-bench-101-data/nasbench_101_cell_list_7.pkl', 'rb') as f:
        cell_list = pickle.load(f)

    print(calculate_dataset_level(ModelSpec(cell_list[10000][0], cell_list[10000][1]), input_shape, data))
