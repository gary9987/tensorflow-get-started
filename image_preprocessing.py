from matplotlib import pyplot as plt
from tensorflow.keras import layers
import PIL.Image
import tensorflow as tf
import pathlib


def distort_color_img(image):
    #image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.image.random_brightness(image, max_delta=32. / 255.)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    image = tf.clip_by_value(image, 0.0, 255)
    """
    Scale to [-1, 1] instead of [0, 1)
    The following code need to be unmarked if really in used.
    """
    #image = tf.math.multiply(image, 1. / 255)
    #image = tf.math.subtract(image, 0.5)
    #image = tf.math.multiply(image, 2.0)
    return image


def distort_color():
    return layers.Lambda(lambda x: distort_color_img(x))


class DistortColor(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.distort_color = distort_color()
    def call(self, x):
        return self.distort_color(x)


def train_preprocessing():
    """
    Training preprocessing layer
    :return:
    """
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        DistortColor()
    ])


def crop_and_resize_image(image):
    height, width = image.shape[0], image.shape[1]
    #print(image.shape)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    image = tf.image.central_crop(image, central_fraction=0.875)

    # Resize the image to the original height and width.
    image = tf.expand_dims(image, 0)
    image = tf.compat.v1.image.resize_bilinear(image, [height, width],
                                     align_corners=False)
    image = tf.squeeze(image, [0])
    return image


def crop_and_resize():
    return layers.Lambda(lambda x: crop_and_resize_image(x))


class CropAndResize(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.crop_and_resize = crop_and_resize()
    def call(self, x):
        return self.crop_and_resize(x)

def eval_preprocessing():
    """
    Evaluating preprocessing layer
    :return:
    """
    return tf.keras.Sequential([
        CropAndResize()
    ])


if __name__ == '__main__':

    data_dir = pathlib.Path('./jpg')

    image_list = list(data_dir.glob('**/*.jpg'))
    img0 = PIL.Image.open(str(image_list[0]))

    train_img = DistortColor()(tf.keras.preprocessing.image.img_to_array(img0))
    plt.imshow(train_img.numpy().astype("uint8"))
    plt.show()

    eval_img = eval_preprocessing()(tf.keras.preprocessing.image.img_to_array(img0))
    plt.imshow(eval_img.numpy().astype("uint8"))
    plt.show()