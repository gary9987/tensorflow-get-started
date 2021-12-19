import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from model import Classifier, CustomModel
import numpy as np

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

    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'mnist',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    num_classes = metadata.features['label'].num_classes

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

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam()

    ori_model = CustomModel()
    # ori_model.build((None, 28, 28, 1))
    # print(ori_model.summary())
    model = tf.keras.Sequential()

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, mode='min')
    model.compile(optimizer=optimizer,
                  loss=loss_object,
                  metrics=['accuracy'])

    epochs = 20
    look_ahead_epochs = 2
    for ind in range(len(ori_model.layers)):

        # if (ori_model.layers[ind].name[0] == 'c'):
        if True:
            # freeze the pre layer for look-ahead process
            for i in range(ind):
                model.layers[i].trainable = False
                # print(model.layers[i].name + ' False')

            # Add k+1 sublayer
            model.add(ori_model.layers[ind])
            # Add classifier
            model.add(tf.keras.Sequential([Classifier()]))

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
            for i in range(ind + 1):
                model.layers[i].trainable = True

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[early_stopping_callback]
            )

            # Pop the classifier
            if ind + 1 != len(ori_model.layers):
                model = tf.keras.models.Sequential(model.layers[:-1])

    model.evaluate(test_ds, verbose=2)
