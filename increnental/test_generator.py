import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from model import Classifier, CustomModelForTest, CustomInceptionModel, CustomBranch
import numpy as np
from keras.callbacks import CSVLogger
from model_generator import model_generator
import csv
from pathlib import Path
import hashlib
from arch_generator import generate_arch

batch_size = 128
AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, data_augmentation=None, shuffle=False, augment=False):
    # Resize and rescale all datasets.
    # ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


if __name__ == '__main__':

    dataset_name = 'mnist'
    (train_ds, val_ds, test_ds), metadata = tfds.load(
        dataset_name,
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    Path("./log").mkdir(parents=True, exist_ok=True)
    with open('./log/' + dataset_name+'.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow(['Architecture', 'LogFilename'])

    log_content = [['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss']]

    data_augmentation = tf.keras.Sequential([
        layers.Rescaling(1. / 255),
        layers.RandomRotation(0.2),
        layers.CenterCrop(28, 28)
    ])

    valid_augmentation = tf.keras.Sequential([
        layers.Rescaling(1. / 255),
        layers.CenterCrop(28, 28)
    ])

    train_ds = prepare(train_ds, data_augmentation, shuffle=True, augment=True)
    val_ds = prepare(val_ds, valid_augmentation, augment=True)
    test_ds = prepare(test_ds, valid_augmentation, augment=True)

    #train_ds = train_ds.cache()
    #val_ds = val_ds.cache()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    model_par = generate_arch(amount_of_cell_layers=1, start=0, end=0)[0]

    ori_model = model_generator(model_par)
    ori_model.build([None, 28, 28, 1])

    for layer_no in range(len(ori_model.layers)):
        print(ori_model.layers[layer_no].name)

    print(ori_model.summary())

    model = tf.keras.Sequential()

    # TODO define patience
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
    model.compile(optimizer=optimizer,
                  loss=loss_object,
                  metrics=['accuracy'])

    epochs = 20
    look_ahead_epochs = 1
    # layer_no start from 0 which is the first layer
    layer_no = 0

    while layer_no < len(ori_model.layers):
        print(layer_no)
        # if (ori_model.layers[ind].name[0] == 'c'):
        if True:
            # freeze the pre layer for look-ahead process
            for i in range(layer_no):
                model.layers[i].trainable = False
                # print(model.layers[i].name + ' False')

            # Add k+1 sublayer
            model.add(ori_model.layers[layer_no])
            # Skip when meet a pooling layer
            while layer_no + 1 < len(ori_model.layers) and ('pool' in ori_model.layers[layer_no + 1].name or
                                                            'drop' in ori_model.layers[layer_no + 1].name or
                                                            'activation' in ori_model.layers[layer_no + 1].name):
                print(ori_model.layers[layer_no + 1].name, ' layer is not trainable so it will be added')
                layer_no += 1
                model.add(ori_model.layers[layer_no])

            # Add classifier
            model.add(tf.keras.Sequential([Classifier(10)]))

            for i in model.layers:
                print(i.name, 'trainable:', i.trainable)

            model.compile(optimizer=optimizer,
                          loss=loss_object,
                          metrics=['accuracy'])

            # train a epochs for Look-Ahead phase
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=look_ahead_epochs
            )

            # train
            for i in range(layer_no + 1):
                model.layers[i].trainable = True

            arch_hash = hashlib.shake_128(str(model_par[:layer_no+1]).encode('utf-8')).hexdigest(10)
            csv_logger_callback = CSVLogger('./log/'+arch_hash+'.csv', append=False, separator=',')
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[early_stopping_callback, csv_logger_callback]
            )

            print(model.summary())

            #  ====================log process==============================
            # Append log of this time to log_content.
            with open('./log/'+arch_hash+'.csv', 'r', newline='') as csvfile:
                rows = csv.reader(csvfile)
                for i in rows:
                    if i[0] == 'epoch':
                        continue
                    log_content.append(i)

            # Write whole log_content to log file.
            with open('./log/'+arch_hash+'.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(log_content)

            # Write the match between architecture and log file name to dataset log file.
            with open('./log/' + dataset_name + '.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([str(model_par[:layer_no+1]), arch_hash + '.csv'])
            # ==============================================================

            # Pop the classifier
            if layer_no + 1 != len(ori_model.layers):
                model = tf.keras.models.Sequential(model.layers[:-1])

            layer_no += 1

    model.evaluate(test_ds, verbose=2)
