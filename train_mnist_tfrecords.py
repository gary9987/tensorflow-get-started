import tensorflow as tf
import tensorflow_datasets as tfds
from dataset_util import DatasetUtil
from tensorflow.keras import layers

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
    print(num_classes)

    train_path = './output/train.tfrecords'
    val_path = './output/val.tfrecords'
    test_path = './output/test.tfrecords'

    DatasetUtil(train_ds).to_tfrecords(train_path)
    DatasetUtil(val_ds).to_tfrecords(val_path)
    DatasetUtil(test_ds).to_tfrecords(test_path)

    train_ds = DatasetUtil(shape=[28, 28, 1]).from_tfrecords(train_path)
    val_ds = DatasetUtil(shape=[28, 28, 1]).from_tfrecords(val_path)
    test_ds = DatasetUtil(shape=[28, 28, 1]).from_tfrecords(test_path)

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

    # Create an instance of the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(8, 3, activation='relu'),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # img, l = next(iter(train_ds))

    model.compile(optimizer=optimizer,
                  loss=loss_object,
                  metrics=['accuracy'])

    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.evaluate(test_ds, verbose=2)
