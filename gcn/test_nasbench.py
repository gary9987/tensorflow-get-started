import os.path
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score, accuracy_score, confusion_matrix
import keras.models
import numpy as np
from spektral.data import BatchLoader
from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
from train_nasbench import get_weighted_mse_loss_func


def test_method(weight_path):
    log_path = f'{weight_path}_test.log'
    if os.path.exists(log_path):
        os.remove(log_path)

    logging.basicConfig(filename=log_path, level=logging.INFO, force=True)
    batch_size = 64
    weight_alpha = 1

    model = keras.models.load_model(weight_path,
                                    custom_objects={'weighted_mse': get_weighted_mse_loss_func(80, weight_alpha)})
    model.compile('adam', loss=get_weighted_mse_loss_func(mid_point=80, alpha=weight_alpha))

    test_dataset = NasBench101Dataset(start=145001, end=169593, preprocessed=True)  # 145001 169593

    test_dataset.apply(NormalizeParAndFlop_NasBench101())
    test_dataset.apply(RemoveTrainingTime_NasBench101())
    test_dataset.apply(Normalize_x_10to15_NasBench101())
    test_dataset.apply(NormalizeLayer_NasBench101())
    test_dataset.apply(LabelScale_NasBench101())
    test_dataset.apply(NormalizeEdgeFeature_NasBench101())
    if 'ecc_con' not in weight_path:
        test_dataset.apply(RemoveEdgeFeature_NasBench101())
    test_dataset.apply(SelectNoneNanData_NasBench101())

    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test loss: {}'.format(loss))
    logging.info('Test loss: {}'.format(loss))
    # Test loss: [0.00380403408780694, 0.00380403408780694]

    # delta <= 80%, > 80%
    delta = [{}, {}]
    for i in range(-11, 12):
        delta[0][i] = 0
        delta[1][i] = 0

    label_array = np.array([])
    pred_array = np.array([])

    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            # logging.info(f'{i} {j}')
            valid_label, valid_predict = i[1], j[1]
            diff = valid_predict - valid_label
            try:
                if valid_label <= 80:
                    delta[0][int(diff / 10)] += 1
                else:
                    delta[1][int(diff / 10)] += 1
            except:
                logging.info(f'Data out of range label: {valid_label}, pred: {valid_predict}')

            label_array = np.concatenate((label_array, np.array(1 if valid_label > 80 else 0)), axis=None)
            pred_array = np.concatenate((pred_array, np.array(1 if valid_predict > 80 else 0)), axis=None)

    for i in range(2):
        if i == 0:
            logging.info('Below information of diff range is accuracy <= 80%')
        else:
            logging.info('Below information diff range is accuracy > 80%')
        for key in delta[i]:
            logging.info(f'Diff range {key * 10}~{key * 10 + 9}: {delta[i][key]}')

    logging.info(f'Confuse matrix: \n{confusion_matrix(label_array, pred_array)}')
    for i in range(2):
        logging.info(f'F1-Score for class-{i}: \n{f1_score(label_array, pred_array, pos_label=i)}')
    logging.info(f'Recall(Sensitivity)-Score: \n{recall_score(label_array, pred_array)}')
    logging.info(f'Specificity-Score: \n{recall_score(label_array, pred_array, pos_label=0)}')
    logging.info(f'Accuracy: \n{accuracy_score(label_array, pred_array)}')
    logging.info(f'Balanced-Accuracy: \n{balanced_accuracy_score(label_array, pred_array)}')

    '''
    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    logging.info('******************************************************************************')
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            if i[0] <= 80:
                logging.info(f'{i} {j}')
    '''


def is_weight_dir(filename):
    check_list = ['ecc_conv', 'gin_conv', 'gat_conv']
    for i in check_list:
        if i in filename:
            return True

    return False


if __name__ == '__main__':
    for filename in os.listdir():
        if os.path.isdir(filename) and is_weight_dir(filename):
            print(f'Now test {filename}')
            test_method(filename)
