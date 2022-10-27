import keras.models
from spektral.data import BatchLoader
from nas_bench_101_dataset import NasBench101Dataset
from transformation import *
import logging
from train_nasbench import get_weighted_mse_loss_func


if __name__ == '__main__':
    logging.basicConfig(filename='test.log', level=logging.INFO, force=True)

    batch_size = 64
    weight_alpha = 1

    model = keras.models.load_model('weights', custom_objects={'weighted_mse': get_weighted_mse_loss_func(80, weight_alpha)})
    model.compile('adam', loss=get_weighted_mse_loss_func(mid_point=80, alpha=weight_alpha))

    test_dataset = NasBench101Dataset(start=120001, end=160000, preprocessed=True)  # 80000 80250

    test_dataset.apply(NormalizeParAndFlop_NasBench101())
    test_dataset.apply(RemoveTrainingTime_NasBench101())
    test_dataset.apply(Normalize_x_10to15_NasBench101())
    test_dataset.apply(NormalizeLayer_NasBench101())
    test_dataset.apply(LabelScale_NasBench101())
    test_dataset.apply(NormalizeEdgeFeature_NasBench101())
    test_dataset.apply(RemoveEdgeFeature_NasBench101())
    test_dataset.apply(SelectNoneNanData_NasBench101())

    print(test_dataset)

    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    loss = model.evaluate(test_loader.load(), steps=test_loader.steps_per_epoch)
    print('Test loss: {}'.format(loss))
    logging.info('Test loss: {}'.format(loss))
    # Test loss: [0.00380403408780694, 0.00380403408780694]

    delta = [0] * 11

    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            logging.info(f'{i} {j}')

            valid_label, valid_predict = i[1], j[1]
            diff = abs(valid_predict - valid_label)
            delta[diff // 10] += 1

    for i, j in enumerate(delta):
        logging.info(f'Diff range {i * 10}~{i * 10 + 9}: {j}')

    test_loader = BatchLoader(test_dataset, batch_size=batch_size, shuffle=False, epochs=1)
    logging.info('******************************************************************************')
    for data in test_loader:
        pred = model.predict(data[0])
        for i, j in zip(data[1], pred):
            if i[0] <= 80:
                logging.info(f'{i} {j}')
