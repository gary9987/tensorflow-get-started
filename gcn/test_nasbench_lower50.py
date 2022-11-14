import os
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, accuracy_score, confusion_matrix
import keras.models
import numpy as np
from spektral.data import BatchLoader
from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
from nasbench_model import get_weighted_mse_loss_func, is_weight_dir
from test_nasbench_metric import *

def test_method(log_dir, weight_path, mid_point):
    # log_path = f'test_result/gin_conv_batch_filterTrue_mp{mid_point}_a1_r1_m256_b128_dropout0.2_lr0.001_mlp(64, 64, 64, 64)_test.log'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_path = os.path.join(log_dir, f'{Path(weight_path).name}_test.log')

    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)
    batch_size = 64
    weight_alpha = 1

    model = keras.models.load_model(weight_path,
                                    custom_objects={
                                        'weighted_mse': get_weighted_mse_loss_func(mid_point, weight_alpha)})

    test_datasets = [
        NasBench101Dataset(start=174801, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True),
        NasBench101Dataset(start=174801, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True,
                           mid_point=mid_point / 100, request_lower=True),
        NasBench101Dataset(start=174801, end=194617, matrix_size_list=[3, 4, 5, 6, 7], preprocessed=True,
                           mid_point=mid_point / 100, request_lower=False)
    ]  # 145001 169593
    print(test_datasets)

    for test_dataset in test_datasets:
        test_dataset.apply(NormalizeParAndFlop_NasBench101())
        test_dataset.apply(RemoveTrainingTime_NasBench101())
        test_dataset.apply(Normalize_x_10to15_NasBench101())
        test_dataset.apply(NormalizeLayer_NasBench101())
        test_dataset.apply(LabelScale_NasBench101())
        test_dataset.apply(NormalizeEdgeFeature_NasBench101())
        if 'ecc_con' not in weight_path:
            test_dataset.apply(RemoveEdgeFeature_NasBench101())
        test_dataset.apply(SelectNoneNanData_NasBench101())

    test_loader = BatchLoader(test_datasets[0], batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test loss: {}'.format(loss))
    logging.info('Test loss: {}'.format(loss))

    # delta <= mid_point%, > mid_point80%
    delta = [{}, {}]
    for i in range(-11, 12):
        delta[0][i] = 0
        delta[1][i] = 0

    label_array_bin = np.array([])
    pred_array_bin = np.array([])
    label_array = np.array([])
    pred_array = np.array([])

    test_loader = BatchLoader(test_datasets[0], batch_size=batch_size, shuffle=False, epochs=1)
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            # logging.info(f'{i} {j}')
            valid_label, valid_predict = i[1], j[1]
            diff = valid_predict - valid_label
            try:
                if valid_label <= mid_point:
                    delta[0][int(diff / 10)] += 1
                else:
                    delta[1][int(diff / 10)] += 1
            except:
                logging.info(f'Data out of range label: {valid_label}, pred: {valid_predict}')

            label_array_bin = np.concatenate((label_array_bin, np.array(0 if valid_label <= mid_point else 1)), axis=None)
            pred_array_bin = np.concatenate((pred_array_bin, np.array(0 if valid_predict <= mid_point else 1)), axis=None)
            label_array = np.concatenate((label_array, np.array(valid_label)), axis=None)
            pred_array = np.concatenate((pred_array, np.array(valid_predict)), axis=None)

    for i in range(2):
        if i == 0:
            logging.info(f'Below information of diff range is accuracy <= {mid_point}%')
        else:
            logging.info(f'Below information diff range is accuracy > {mid_point}%')
        for key in delta[i]:
            logging.info(f'Diff range {key * 10}~{key * 10 + 9}: {delta[i][key]}')

    logging.info(f'Confuse matrix: \n{confusion_matrix(label_array_bin, pred_array_bin)}')
    for i in range(2):
        logging.info(f'F1-Score for class-{i}: \n{f1_score(label_array_bin, pred_array_bin, pos_label=i)}')
    logging.info(f'Recall(Sensitivity)-Score: \n{recall_score(label_array_bin, pred_array_bin)}')
    logging.info(f'Specificity-Score: \n{recall_score(label_array_bin, pred_array_bin, pos_label=0)}')
    logging.info(f'Accuracy: \n{accuracy_score(label_array_bin, pred_array_bin)}')
    logging.info(f'Balanced-Accuracy: \n{balanced_accuracy_score(label_array_bin, pred_array_bin)}')

    test_loader = BatchLoader(test_datasets[1], batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test MSE loss for lower split: {}'.format(loss))
    logging.info('Test MSE loss for lower split: {}'.format(loss))

    test_loader = BatchLoader(test_datasets[2], batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test MSE loss for upper split: {}'.format(loss))
    logging.info('Test MSE loss for upper split: {}'.format(loss))

    model.compile('adam', loss='mae')
    test_loader = BatchLoader(test_datasets[1], batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test MAE loss for lower split: {}'.format(loss))
    logging.info('Test MAE loss for lower split: {}'.format(loss))

    test_loader = BatchLoader(test_datasets[2], batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test MAE loss for upper split: {}'.format(loss))
    logging.info('Test MAE loss for upper split: {}'.format(loss))

    test_loader = BatchLoader(test_datasets[0], batch_size=batch_size, shuffle=False, epochs=1)
    all_mae_loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)

    num_select = 100
    num_judge = 50

    test_count = 1000
    mis_count = 0
    kt_list = []
    p_list = []

    for _ in range(test_count):
        pred_list, label_list = randon_select_data(pred_array, label_array, mid_point, num_select, 1, num_judge, minor_bound=50)
        kt, p = kendalltau(pred_list, label_list)
        kt_list.append(kt)
        p_list.append(p)
        if is_misjudgment(pred_list, label_list, mid_point, num_select, num_judge):
            mis_count += 1

    logging.info(f'The misjudgement ratio: {(mis_count / test_count) * 100}%')
    logging.info(f'Avg KT rank correlation: {sum(kt_list) / len(kt_list)}')
    logging.info(f'Avg P value: {sum(p_list) / len(p_list)}')
    logging.info(f'Std KT rank correlation: {np.std(kt_list)}')
    logging.info(f'Std P value: {np.std(p_list)}')

    print('Test MAE loss for all split: {}'.format(all_mae_loss))
    logging.info('Test MAE loss for all split: {}'.format(all_mae_loss))


if __name__ == '__main__':

    log_dir = 'test_full_model_lower50_result'
    model_dir = 'full_model'

    for filename in os.listdir(model_dir):
        if os.path.isdir(os.path.join(model_dir, filename)) and is_weight_dir(os.path.join(model_dir, filename)):
            print(f'Now test {os.path.join(model_dir, filename)}')
            mp_pos = filename.find('mp')
            mid_point = int(filename[mp_pos+2: mp_pos+4])
            test_method(log_dir, os.path.join(model_dir, filename), mid_point)
    '''
    for i in range(10, 91, 10):
        print(f'Now mp is {i}')
    '''
    #test_method('full_model/gin_conv_batch_filterTrue_mp90_a1.5_r1_m256_b128_dropout0.2_lr0.001_mlp(64, 64, 64, 64)', 90)