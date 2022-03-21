import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import random
from model import Classifier
import numpy as np
from keras.callbacks import CSVLogger
import csv
from pathlib import Path
import hashlib
from argparse import ArgumentParser, Namespace
from model_builder import build_arch_model
from model_spec import ModelSpec
from os import path
from augmentation import Augmentation


def prepare(ds, seed, data_augmentation=None, shuffle=False, augment=False, batch_size=128, autotune=tf.data.AUTOTUNE):
    # Resize and rescale all datasets.
    # ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=1000, seed=seed)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=autotune)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=autotune).cache()


class LrCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, amount, batch_size, total_layers, optimizer):
        super(LrCustomCallback, self).__init__()
        self.global_batch = 0
        self.optimizer = optimizer
        self.total_batches = int(total_layers * args.epochs * amount / batch_size)

    def on_train_batch_end(self, batch, logs=None):
        self.global_batch += 1
        progress_fraction = self.global_batch / self.total_batches
        learning_rate = (0.5 * args.lr * (1 + tf.cos(np.pi * progress_fraction)))
        tf.keras.backend.set_value(self.optimizer.lr, learning_rate)

    def on_epoch_begin(self, epoch, logs=None):
        print('Learning Rate: ', tf.keras.backend.eval(self.optimizer.lr))


def incremental_training(args, cell_filename: str, start=0, end=0):
    """
    :param args:
    :param cell_filename:
    :param start: The start cell index want to train
    :param end: The end cell index want to train include end index
    """

    # ==========================================================
    # Setting some parameters here.
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    AUTOTUNE = tf.data.AUTOTUNE
    log_path = './' + dataset_name + '_log/'
    epochs = args.epochs
    look_ahead_epochs = args.look_ahead_epochs
    inputs_shape = args.inputs_shape
    tf.keras.utils.set_random_seed(args.seed)
    # ==========================================================

    (train_ds, val_ds, test_ds), metadata = tfds.load(
        dataset_name,
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    num_classes = metadata.features['label'].num_classes

    # The dict is to record amount of the times which the arch appear
    arch_count_map = dict()

    Path(log_path).mkdir(parents=True, exist_ok=True)
    pkl_count = 0
    while path.exists(log_path + dataset_name + '_' + str(pkl_count) + '.pkl'):
        pkl_count += 1
    pkl_path = log_path + dataset_name + '_' + str(pkl_count) + '.pkl'
    with open(pkl_path, 'wb') as file:
        pickle.dump([], file)

    augmentation = Augmentation(args.seed)
    train_ds = prepare(train_ds, args.seed, augmentation.get_augmentation('cifar10', True), shuffle=False, augment=True, batch_size=batch_size,
                       autotune=AUTOTUNE)
    val_ds = prepare(val_ds, args.seed, augmentation.get_augmentation('cifar10', False), augment=False, batch_size=batch_size, autotune=AUTOTUNE)

    # cache the dataset on memory
    # 2022/03/20 Update cache() move to prepare() function
    # train_ds = train_ds.cache()
    # val_ds = val_ds.cache()

    file = open(cell_filename, 'rb')
    cell_list = pickle.load(file)
    file.close()
    random.seed(args.seed)
    random.shuffle(cell_list)

    for cell in cell_list[start: end + 1]:
        print('Running on Cell:', cell)
        # log content will store the training records of every architecture.
        log_content = [['epoch', 'accuracy', 'loss', 'val_accuracy', 'val_loss']]

        matrix, ops = cell[0], cell[1]

        spec = ModelSpec(np.array(matrix), ops)
        ori_model = build_arch_model(spec, inputs_shape)
        ori_model.build([*inputs_shape])

        for layer_no in range(len(ori_model.layers)):
            print(ori_model.layers[layer_no].name)

        print(ori_model.summary())

        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=args.lr, momentum=args.momentum, epsilon=1.0)
        lr_scheduler_callback = LrCustomCallback(metadata.splits['train'].num_examples,
                                                 args.batch_size,
                                                 len(ori_model.layers),
                                                 optimizer)

        model = tf.keras.Sequential()

        # layer_no start from 0 which is the first layer
        layer_no = 0
        while layer_no < len(ori_model.layers):

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
            model.add(tf.keras.Sequential([Classifier(num_classes, spec.data_format)]))

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
            print("Look-Ahead Finished.")
            # print(model.summary())
            # train
            for i in range(layer_no + 1):
                model.layers[i].trainable = True

            model.compile(optimizer=optimizer,
                          loss=loss_object,
                          metrics=['accuracy'])

            # print(model.summary())
            arch_hash = hashlib.shake_128((str(matrix) + str(ops) + str(layer_no)).encode('utf-8')).hexdigest(10)
            arch_count = 0
            if arch_count_map.get(arch_hash) is not None:
                arch_count_map[arch_hash] = arch_count_map[arch_hash] + 1
                arch_count = arch_count_map.get(arch_hash)
            else:
                arch_count_map[arch_hash] = 0
            arch_hash += '_' + str(arch_count)

            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=args.patience,
                                                                       mode='min')
            csv_logger_callback = CSVLogger(log_path + arch_hash + '.csv', append=False, separator=',')
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[early_stopping_callback, csv_logger_callback, lr_scheduler_callback]
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
            pkl_file = open(pkl_path, 'rb')
            pkl_log = pickle.load(pkl_file)
            pkl_file.close()
            pkl_log.append({'matrix': matrix, 'ops': ops, 'layers': layer_no, 'log_file': arch_hash + '.csv'})
            pkl_file = open(pkl_path, 'wb')
            pickle.dump(pkl_log, pkl_file)
            pkl_file.close()
            # ==============================================================

            # Pop the classifier
            if layer_no + 1 != len(ori_model.layers):
                model = tf.keras.models.Sequential(model.layers[:-1])

            layer_no += 1


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    # train
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--look_ahead_epochs", type=int, default=1)
    # TODO change patience
    parser.add_argument("--patience", type=int, default=3)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)

    # data
    parser.add_argument("--dataset_name", type=str, default='cifar10')
    parser.add_argument("--batch_size", type=int, default=256)
    # IMPORTANT: inputs_shape need to match with dataset
    parser.add_argument("--inputs_shape", type=tuple, default=(None, 32, 32, 3))

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    incremental_training(args, cell_filename='./cell_list.pkl', start=0, end=0)
