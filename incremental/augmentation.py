import tensorflow as tf
from tensorflow.keras import layers


class Augmentation:
    def __init__(self, seed):
        self.seed = seed

    def get_augmentation(self, name, training=True):
        if name == 'cifar10':
            return self.cifar10(training)
        if name == 'cifar100':
            return self.cifar100(training)
        if name == 'mnist':
            return self.mnist(training)

    def cifar10(self, training):
        rgb_mean = [0.4914, 0.4822, 0.4465]
        rgb_std = [0.247, 0.243, 0.261]
        if training:
            return tf.keras.Sequential([
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Lambda(lambda x: x - tf.constant(rgb_mean, shape=[1, 1, 3])),
                tf.keras.layers.Lambda(lambda x: x / tf.constant(rgb_std, shape=[1, 1, 3])),
                tf.keras.layers.ZeroPadding2D(padding=4),
                tf.keras.layers.RandomCrop(height=32, width=32, seed=self.seed),
                tf.keras.layers.RandomFlip(mode='horizontal', seed=self.seed),
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Lambda(lambda x: x - tf.constant(rgb_mean, shape=[1, 1, 3])),
                tf.keras.layers.Lambda(lambda x: x / tf.constant(rgb_std, shape=[1, 1, 3])),
            ])

    def cifar100(self, training):
        rgb_mean = [0.5071, 0.4867, 0.4408]
        rgb_std = [0.2675, 0.2565, 0.2761]
        if training:
            return tf.keras.Sequential([
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Lambda(lambda x: x - tf.constant(rgb_mean, shape=[1, 1, 3])),
                tf.keras.layers.Lambda(lambda x: x / tf.constant(rgb_std, shape=[1, 1, 3])),
                tf.keras.layers.ZeroPadding2D(padding=4),
                tf.keras.layers.RandomCrop(height=32, width=32, seed=self.seed),
                tf.keras.layers.RandomFlip(mode='horizontal', seed=self.seed),
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Lambda(lambda x: x - tf.constant(rgb_mean, shape=[1, 1, 3])),
                tf.keras.layers.Lambda(lambda x: x / tf.constant(rgb_std, shape=[1, 1, 3])),
            ])

    def mnist(self, training):
        rgb_mean = [0.1307]
        rgb_std = [0.3081]
        if training:
            return tf.keras.Sequential([
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Lambda(lambda x: x - tf.constant(rgb_mean, shape=[1, 1, 1])),
                tf.keras.layers.Lambda(lambda x: x / tf.constant(rgb_std, shape=[1, 1, 1])),
                tf.keras.layers.ZeroPadding2D(padding=1),
                tf.keras.layers.RandomCrop(height=30, width=30, seed=self.seed),
                tf.keras.layers.RandomFlip(mode='horizontal', seed=self.seed),
            ])
        else:
            return tf.keras.Sequential([
                tf.keras.layers.Rescaling(1. / 255),
                tf.keras.layers.Lambda(lambda x: x - tf.constant(rgb_mean, shape=[1, 1, 1])),
                tf.keras.layers.Lambda(lambda x: x / tf.constant(rgb_std, shape=[1, 1, 1])),
            ])