import logging
import random
from spektral.data import Dataset, Graph
import pickle
import numpy as np
from model_spec import ModelSpec
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
import os
import wget
from os import path
import re
import hashlib
import model_builder
from keras import backend as K
from classifier import Classifier
import model_util
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from tensorflow.python.profiler.model_analyzer import profile
import csv
from nas_bench_101_dataset import train_valid_test_split_dataset


logging.basicConfig(filename='nas_bench_101_dataset_partial.log', level=logging.INFO)


class NasBench101DatasetPartial(Dataset):
    def __init__(self, start: int, end: int, size: int, matrix_size_list: list, select_seed: int,
                 shuffle_seed=0, shuffle=True, preprocessed=False, **kwargs):

        self.nodes = 67
        self.features_dict = {'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4,
                              'Classifier': 5, 'maxpool2x2': 6, 'flops': 7, 'params': 8, 'num_layer': 9,
                              'input_shape_1': 10, 'input_shape_2': 11, 'input_shape_3': 12, 'output_shape_1': 13,
                              'output_shape_2': 14, 'output_shape_3': 15}

        self.num_features = len(self.features_dict)
        self.preprocessed = preprocessed
        if preprocessed:
            self.file_path_prefix = 'Preprocessed_NasBench101Dataset'
            self.file_path_suffix = 'Preprocessed_NasBench101Dataset_'
        else:
            self.file_path_prefix = 'NasBench101Dataset'
            self.file_path_suffix = 'NasBench101Dataset_'

        self.select_seed = select_seed
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.total_layers = 11
        self.matrix_size_list = matrix_size_list
        self.start = start
        self.end = end
        self.size = size

        super().__init__(**kwargs)


    def download(self):
        if not os.path.exists(self.file_path_prefix):
            print('Downloading...')
            if self.preprocessed:
                file_name = wget.download('https://www.dropbox.com/s/muetcgm9l1e01mc/Preprocessed_NasBench101Dataset.zip?dl=1')
            else:
                file_name = wget.download('https://www.dropbox.com/s/40lrvb3lcgij5c8/NasBench101Dataset.zip?dl=1')
            print('Save dataset to {}'.format(file_name))
            os.system('unzip {}'.format(file_name))
            print(f'Unzip dataset finish.')


    def read(self):

        output = []
        filename_list = []
        matrix_size_list = self.matrix_size_list

        for size in matrix_size_list:
            path = self.file_path_prefix + '/' + self.file_path_suffix + f'{size}'
            for i in range(len(os.listdir(path))):
                #with np.load(os.path.join(path, f'graph_{i}.npz')) as npz:
                #    data = {'x': npz['x'], 'e': npz['e'], 'a': npz['a'], 'y': npz['y']}
                filename_list.append(os.path.join(path, f'graph_{i}.npz'))

        if self.shuffle:
            random.seed(self.shuffle_seed)
            random.shuffle(filename_list)

        filename_list = filename_list[self.start: self.end]
        if self.size >= len(filename_list):
            raise Exception('size is greater than filename_list')

        count = 0
        random.seed(self.select_seed)

        visited = set()
        while count < self.size:
            filename = random.choice(filename_list)
            if filename in visited:
                continue

            visited.add(filename)
            graph_data = np.load(filename)

            if self.preprocessed:
                if np.isnan(graph_data['y'][0][0]) and np.isnan(graph_data['y'][1][0]) and np.isnan(graph_data['y'][2][0]):
                    continue

            output.append(Graph(x=graph_data['x'], e=graph_data['e'], a=graph_data['a'], y=graph_data['y']))
            count += 1

        return output


if __name__ == '__main__':
    data = NasBench101DatasetPartial(start=0, end=155000, size=500, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True)
    # Train/valid/test split
    data_dict = train_valid_test_split_dataset(data, ratio=[0.8, 0.15, 0.05])
    pass
