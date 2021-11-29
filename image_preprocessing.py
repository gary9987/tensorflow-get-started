import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import layers
import keras
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import pathlib

def distort_color_img(image):
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    return image

def distort_color():
  return layers.Lambda(lambda x: distort_color_img(x))


def train_preprocessing():
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        distort_color(),
    ])
    return augmentation


if __name__ == '__main__':

    distort_color = distort_color()

    data_dir = pathlib.Path('./jpg')

    image_list = list(data_dir.glob('**/*.jpg'))
    img0 = PIL.Image.open(str(image_list[0]))


    augmented_image = distort_color(img0)
    plt.imshow(augmented_image.numpy().astype("uint8"))
    plt.show()