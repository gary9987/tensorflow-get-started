import os.path
from pathlib import Path
import logging
from scipy.stats import kendalltau
from test_nasbench_metric import randon_select_data, mAP
from sklearn.metrics import ndcg_score, mean_squared_error, mean_absolute_error
import numpy as np


def test_metric_rnn(log_dir, weight_path, test_dataset, model):

    if not os.path.exists(log_dir):
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    log_path = os.path.join(log_dir, f'{Path(weight_path).name}_test.log')

    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)

    #model = XGBRegressor()
    #model.load_model(weight_path)

    pred = model.predict(test_dataset['x'], batch_size=256)
    mse = np.sqrt(mean_squared_error(test_dataset['y'], pred))
    print('Test MSE loss: {}'.format(mse))
    logging.info('Test MSE loss: {}'.format(mse))

    mae = np.sqrt(mean_absolute_error(test_dataset['y'], pred))
    print('Test MAE loss: {}'.format(mae))
    logging.info('Test MAE loss: {}'.format(mae))

    label_array = np.array([])
    pred_array = np.array([])

    for valid_label, valid_predict in zip(test_dataset['y'], pred):
        # logging.info(f'{i} {j}')
        label_array = np.concatenate((label_array, np.array(valid_label)), axis=None)
        pred_array = np.concatenate((pred_array, np.array(valid_predict)), axis=None)

    num_select = 100
    test_count = 1000
    kt_list = []
    p_list = []
    mAP_list = []
    ndcg_list = []

    for _ in range(test_count):
        pred_list, label_list = randon_select_data(pred_array, label_array, 0, num_select, 0)
        kt, p = kendalltau(pred_list, label_list)
        kt_list.append(kt)
        p_list.append(p)
        mAP_list.append(mAP(pred_list, label_list, 0.1))
        ndcg_list.append(ndcg_score(np.asarray([label_list]), np.asarray([pred_list])))

    kt = sum(kt_list) / len(kt_list)
    p = sum(p_list) / len(p_list)
    avg_mAP = sum(mAP_list) / len(mAP_list)
    ndcg = sum(ndcg_list) / len(ndcg_list)
    logging.info(f'Avg KT rank correlation: {kt}')
    logging.info(f'Avg P value: {p}')
    logging.info(f'Std KT rank correlation: {np.std(kt_list)}')
    logging.info(f'Std P value: {np.std(p_list)}')
    logging.info(f'Avg mAP value: {avg_mAP}')
    logging.info(f'Std mAP value: {np.std(mAP_list)}')
    logging.info(f'Avg ndcg value: {ndcg}')
    logging.info(f'Std ndcg value: {np.std(ndcg_list)}')

    return {'MSE': mse, 'MAE': mae, 'KT': kt, 'P': p, 'mAP': avg_mAP, 'NDCG': ndcg}
