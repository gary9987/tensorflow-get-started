import tensorflow as tf
from tensorflow.keras import layers


class Augmentation:
    def __init__(self, seed):
        self.seed = seed

    def get_augmentation(self, name, training=True):
        if name == 'cifar10':
            return self.cifar10(training)

    def cifar10(self, training=True):
        if training:
            return tf.keras.Sequential([
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.ZeroPadding2D(padding=4),
                tf.keras.layers.RandomCrop(height=32, width=32, seed=self.seed),
                tf.keras.layers.RandomFlip(mode='horizontal', seed=self.seed),
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Rescaling(1. / 255)
            ])
