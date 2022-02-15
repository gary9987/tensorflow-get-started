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




def prepare(ds, data_augmentation=None, shuffle=False, augment=False, batch_size=128, autotune=tf.data.AUTOTUNE):
    # Resize and rescale all datasets.
    # ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=autotune)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=autotune)


def incremental_training(dataset_name='mnist', amount_of_cell_layers=1, start=0, end=0):
    """
    # ==========================================================
    Setting some parameters here.
    """
    batch_size = 128
    AUTOTUNE = tf.data.AUTOTUNE
    log_path = './' + dataset_name + '_log/'
    epochs = 20
    look_ahead_epochs = 1
    # ==========================================================

    (train_ds, val_ds, test_ds), metadata = tfds.load(
        dataset_name,
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )

    Path(log_path).mkdir(parents=True, exist_ok=True)
    with open(log_path + dataset_name + '.csv', 'w', newline='') as csvfile:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvfile)
        # 寫入一列資料
        writer.writerow(['Architecture', 'LogFilename'])

    # log content will store the training records of every architecture.
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

    train_ds = prepare(train_ds, data_augmentation, shuffle=True, augment=True, batch_size=batch_size, autotune=AUTOTUNE)
    val_ds = prepare(val_ds, valid_augmentation, augment=True, batch_size=batch_size, autotune=AUTOTUNE)
    #test_ds = prepare(test_ds, valid_augmentation, augment=True, batch_size=batch_size, autotune=AUTOTUNE)

    # cache the dataset on memory
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    arch_list = generate_arch(amount_of_cell_layers, start, end)

    for arch in arch_list:
        ori_model = model_generator(arch)
        # TODO The shape need change with different dataset
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

                arch_hash = hashlib.shake_128(str(arch[:layer_no + 1]).encode('utf-8')).hexdigest(10)
                csv_logger_callback = CSVLogger(log_path + arch_hash + '.csv', append=False, separator=',')
                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=epochs,
                    callbacks=[early_stopping_callback, csv_logger_callback]
                )

                print(model.summary())

                #  ====================log process==============================
                # Append log of this time to log_content.
                with open(log_path + arch_hash + '.csv', 'r', newline='') as csvfile:
                    rows = csv.reader(csvfile)
                    for i in rows:
                        if i[0] == 'epoch':
                            continue
                        log_content.append(i)

                # Write whole log_content to log file.
                with open(log_path + arch_hash + '.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(log_content)

                # Write the match between architecture and log file name to dataset log file.
                with open(log_path + dataset_name + '.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([str(arch[:layer_no + 1]), arch_hash + '.csv'])
                # ==============================================================

                # Pop the classifier
                if layer_no + 1 != len(ori_model.layers):
                    model = tf.keras.models.Sequential(model.layers[:-1])

                layer_no += 1

        #model.evaluate(test_ds, verbose=2)


if __name__ == '__main__':
    incremental_training()
