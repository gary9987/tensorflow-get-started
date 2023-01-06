import random
import time
from typing import Union, Tuple, List
import numpy as np
from sklearn.metrics import ndcg_score
from argparse import ArgumentParser
from scipy.stats import kendalltau
from test_nasbench_metric import mAP
from test_utils import download_nas_bench_101_data
from nas_bench_101_dataset import NasBench101Dataset
from transformation import SelectNoneNanData_NasBench101, LabelScale_NasBench101, RemoveEdgeFeature_NasBench101
import logging
import sys


logging.basicConfig(filename=f'test_flop_and_param_by_table.log', force=True, filemode='w')
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def get_dataset():
    #194617
    dataset = NasBench101Dataset(start=0, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True)
    dataset.apply(LabelScale_NasBench101())
    dataset.apply(RemoveEdgeFeature_NasBench101())
    dataset.apply(SelectNoneNanData_NasBench101())
    return dataset


def get_arch_flop_or_param_and_acc_list(sample_datasets: List, get_metric_idx: int) -> Tuple[List, List]:
    """
    :param sample_datasets:
    :param get_metric: can be params, flops
    :return:
    """
    metric_list = []
    valid_acc_list = []

    for data in sample_datasets:
        sum_x = np.sum(data.x, axis=0)
        metric_list.append(sum_x[get_metric_idx])
        val_acc = data.y[1]
        valid_acc_list.append(val_acc)

    return metric_list, valid_acc_list


def get_arch_flop_divided_by_param_and_acc_list(sample_datasets: List, flops_idx: int, params_idx: int) -> Tuple[List, List]:
    metric_list = []
    valid_acc_list = []

    for data in sample_datasets:
        sum_x = np.sum(data.x, axis=0)
        metric_list.append(sum_x[flops_idx] / sum_x[params_idx])
        val_acc = data.y[1]
        valid_acc_list.append(val_acc)

    return metric_list, valid_acc_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--test_count', type=int, default=1000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    download_nas_bench_101_data()
    datasets = get_dataset()

    random.seed(time.time())

    num_select = 100
    test_count = args.test_count

    kt_list = []
    p_list = []
    mAP_list = []
    ndcg_list = []

    for metric in ['flops', 'params', 'divide']:
        for i in range(test_count):
            sample_datasets = random.sample(list(datasets), num_select)
            if metric == 'divide':
                score_list, val_acc_list = get_arch_flop_divided_by_param_and_acc_list(sample_datasets,
                                                                                       datasets.features_dict['flops'],
                                                                                       datasets.features_dict['params'])
            else:
                score_list, val_acc_list = get_arch_flop_or_param_and_acc_list(sample_datasets, datasets.features_dict[metric])

            kt, p = kendalltau(score_list, val_acc_list)
            kt_list.append(kt)
            p_list.append(p)
            mAP_list.append(mAP(score_list, val_acc_list, 0.1))
            ndcg_list.append(ndcg_score(np.asarray([score_list]), np.asarray([val_acc_list])))

        kt = sum(kt_list) / len(kt_list)
        p = sum(p_list) / len(p_list)
        avg_mAP = sum(mAP_list) / len(mAP_list)
        ndcg = sum(ndcg_list) / len(ndcg_list)

        logger.info(f'{metric} result:')
        logger.info(f'Avg KT rank correlation: {kt}')
        logger.info(f'Avg P value: {p}')
        logger.info(f'Std KT rank correlation: {np.std(kt_list)}')
        logger.info(f'Std P value: {np.std(p_list)}')
        logger.info(f'Avg mAP value: {avg_mAP}')
        logger.info(f'Std mAP value: {np.std(mAP_list)}')
        logger.info(f'Avg ndcg value: {ndcg}')
        logger.info(f'Std ndcg value: {np.std(ndcg_list)}')
