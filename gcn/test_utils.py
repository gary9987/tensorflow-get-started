import os
import wget
import pickle
import random


def download_nas_bench_101_data():
    if not os.path.exists('nas-bench-101-data'):
        print('Downloading nas-bench-101-data...')
        file_name = wget.download('https://www.dropbox.com/s/vkexemlekfabxa1/nas-bench-101-data.zip?dl=1')
        print('Save data to {}'.format(file_name))
        os.system('unzip {}'.format(file_name))
        print(f'Unzip data finish.')


def get_all_arch_list(shuffle=False):
    cell_list = []
    for matrix_size in range(3, 4):
        with open(os.path.join('nas-bench-101-data', f'nasbench_101_cell_list_{matrix_size}.pkl'), 'rb') as f:
            cell_list += pickle.load(f)

    if shuffle:
        random.shuffle(cell_list)

    return cell_list
