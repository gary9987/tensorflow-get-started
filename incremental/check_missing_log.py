import csv
import pickle
import tensorflow as tf
import random
from model import Classifier
import numpy as np
import hashlib
from argparse import ArgumentParser, Namespace
from model_builder import build_arch_model
from model_spec import ModelSpec
from os import path
import os
import logging


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    ret = list(mapped_int)
    ret[0] = None
    return tuple(ret)


def check_log_exist(args, cell_filename: str):
    # ==========================================================
    # Setting some parameters here.
    start = args.start
    end = args.end
    inputs_shape = args.inputs_shape
    num_classes = args.num_classes
    log_path = args.log_path
    # ==========================================================
    # Auto download if cell_list.pkl is not exist
    if not path.exists(cell_filename):
        os.system('sh download.sh')
    file = open(cell_filename, 'rb')
    cell_list = pickle.load(file)
    file.close()
    random.shuffle(cell_list)

    missing_list = []
    for now_idx, cell in zip(range(start, end + 1), cell_list[start: end + 1]):

        logging.info('Now running {:d}/{:d}'.format(now_idx, end))
        print('Running on Cell:', cell)

        matrix, ops = cell[0], cell[1]

        spec = ModelSpec(np.array(matrix), ops)
        ori_model = build_arch_model(spec, inputs_shape)
        ori_model.build([*inputs_shape])

        # layer_no start from 0 which is the first layer
        layer_no = 0
        while layer_no < len(ori_model.layers):

            # Skip when meet a pooling layer
            while layer_no + 1 < len(ori_model.layers) and ('pool' in ori_model.layers[layer_no + 1].name or
                                                            'drop' in ori_model.layers[layer_no + 1].name or
                                                            'activation' in ori_model.layers[layer_no + 1].name):
                print(ori_model.layers[layer_no + 1].name, ' layer is not trainable so it will be added')
                layer_no += 1

            # print(model.summary())
            arch_hash = hashlib.shake_128((str(matrix) + str(ops) + str(layer_no)).encode('utf-8')).hexdigest(10)
            arch_count = 0
            arch_hash += '_' + str(arch_count)
            if not os.path.exists(log_path + arch_hash + '.csv'):
                missing_list.append([now_idx, layer_no])

            layer_no += 1

    with open('missing_log.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['index', 'layer_no'])
        writer.writerows(missing_list)

    print(missing_list)


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--log_path", type=str, default='./cifar10_log/')
    parser.add_argument("--num_classes", type=int, default=10)
    # Set start end
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5)

    # IMPORTANT: inputs_shape need to match with dataset
    parser.add_argument("--inputs_shape", type=tuple_type, default="-1, 32, 32, 3")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    check_log_exist(args, cell_filename='cell_list.pkl')