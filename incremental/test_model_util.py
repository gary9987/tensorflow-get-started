import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from model import Classifier
from model_spec import ModelSpec
from model_builder import CellModel, build_arch_model
import numpy as np
from keras.callbacks import CSVLogger
from model_util import get_model_by_id_and_layer

batch_size = 256
AUTOTUNE = tf.data.AUTOTUNE


def prepare(ds, data_augmentation=None, shuffle=False, augment=False):
    # Resize and rescale all datasets.
    #ds = ds.map(lambda x, y: (resize_and_rescale(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(1000)

    # Batch all datasets.
    ds = ds.batch(batch_size)

    # Use data augmentation only on the training set.
    if augment:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)

    # Use buffered prefetching on all datasets.
    return ds.prefetch(buffer_size=AUTOTUNE)


class LrCustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, amount, batch_size, total_layers, optimizer):
        super(LrCustomCallback, self).__init__()
        self.global_batch = 0
        self.optimizer = optimizer
        self.total_batches = int(total_layers * 20 * amount / batch_size)

    def on_train_batch_end(self, batch, logs=None):
        self.global_batch += 1
        progress_fraction = self.global_batch / self.total_batches
        learning_rate = (0.5 * 0.1 * (1 + tf.cos(np.pi * progress_fraction)))
        tf.keras.backend.set_value(self.optimizer.lr, learning_rate)

    def on_epoch_begin(self, epoch, logs=None):
        print('Learning Rate: ', tf.keras.backend.eval(self.optimizer.lr))


if __name__ == '__main__':

    (train_ds, val_ds, test_ds), metadata = tfds.load(
        'mnist',
        split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
        with_info=True,
        as_supervised=True,
    )
    num_classes = metadata.features['label'].num_classes
    print(num_classes)

    data_augmentation = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.RandomRotation(0.2),
        layers.CenterCrop(28, 28)
    ])

    valid_augmentation = tf.keras.Sequential([
        layers.Rescaling(1./255),
        layers.CenterCrop(28, 28)
    ])

    train_ds = prepare(train_ds, data_augmentation, shuffle=True, augment=True)
    val_ds = prepare(val_ds, valid_augmentation, augment=True)
    test_ds = prepare(test_ds, valid_augmentation, augment=True)

    train_ds = train_ds.cache()
    val_ds = val_ds.cache()

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1, momentum=0.9, epsilon=1.0)

    model = get_model_by_id_and_layer('./cell_list.pkl', 0, (None, 28, 28, 1), 0, 5)
    model.add(Classifier(10))
    model.build([None, 28, 28, 1])
    model.summary()

    model.compile(optimizer=optimizer,
              loss=loss_object,
              metrics=['accuracy'])

    epochs = 20 * 12
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min')
    csv_logger_callback = CSVLogger('./normal_training_log.csv', append=False, separator=',')
    lr_scheduler_callback = LrCustomCallback(metadata.splits['train'].num_examples,
                                             batch_size,
                                             12,
                                             optimizer)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping_callback, csv_logger_callback, lr_scheduler_callback]
    )

    model.evaluate(test_ds, verbose=2)