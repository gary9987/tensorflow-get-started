from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras


class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.con1 = tf.keras.layers.Conv2D(8, 3, activation='relu')
        self.con2 = tf.keras.layers.Conv2D(16, 3, activation='relu')
        self.con3 = tf.keras.layers.Conv2D(32, 3, activation='relu')


        self.flatten = tf.keras.layers.Flatten()
        self.den1 = tf.keras.layers.Dense(128, activation='relu')
        self.den2 = tf.keras.layers.Dense(64, activation='relu')
        self.den3 = tf.keras.layers.Dense(10)


    def call(self, inputs):
        x = self.con1(inputs)
        x = self.con2(x)
        x = self.con3(x)
        x = self.flatten(x)
        x = self.den1(x)
        x = self.den2(x)
        x = self.den3(x)
        return x
