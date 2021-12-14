from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras


from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras

class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.con1 = tf.keras.layers.Conv2D(8, 3, activation='relu')
        self.con2 = tf.keras.layers.Conv2D(16, 3, activation='relu')
        self.con3 = tf.keras.layers.Conv2D(32, 3, activation='relu')

    def call(self, inputs):
        x = self.con1(inputs)
        x = self.con2(x)
        x = self.con3(x)
        return x

class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.flt = tf.keras.layers.Flatten()
        self.den1 = tf.keras.layers.Dense(10)

    def call(self, inputs):
        return self.den1(self.flt(inputs))
