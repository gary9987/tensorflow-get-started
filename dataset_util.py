import tensorflow as tf
import os
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


class DatasetUtil:
    def __init__(self, dataset=None, shape=[]):
        """
        :param dataset: give a tfds dataset with supervised
        """
        if dataset is not None:
            self.dataset = dataset.prefetch(tf.data.AUTOTUNE)
            self.dataset = self.dataset.batch(1)

        self.shape = shape

    def to_tfrecords(self, filename='output.tfrecords'):
        """
        :param filename: give a output file path
        """

        if not os.path.exists(os.path.dirname(filename)):
            if os.path.dirname(filename) != '':
                os.makedirs(os.path.dirname(filename))

        with tf.io.TFRecordWriter(filename) as writer:
            for image, label in self.dataset:
                shape = image.shape
                image = image.numpy().tobytes()
                label = int(label)
                feature = {  # create tf.train.Feature dictionary
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # Image is a byte list
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # Label is a int64
                }
                record_bytes = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(record_bytes.SerializeToString())

    def from_tfrecords(self, filename=''):

        raw_dataset = tf.data.TFRecordDataset(filename)
        feature = {  # create tf.train.Feature dictionary
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64)  # Label is a int64
        }

        def __parse_example(example_string):
            feature_dict = tf.io.parse_single_example(example_string, feature)
            img = tf.io.decode_raw(feature_dict['image'], tf.uint8)
            img = tf.reshape(img, self.shape)
            img = tf.cast(img, tf.float32)
            return img, feature_dict['label']

        return raw_dataset.map(__parse_example)


if __name__ == '__main__':
    """
    The example of DatasetUtil usage.
    """
    mnist_dataset, metadata = tfds.load(
        'mnist',
        split='train',
        with_info=True,
        as_supervised=True,
    )

    output_file_path = './output/train.tfrecords'

    # ======== Write tfrecord ========
    # Init class object with tfds dataset
    dataset = DatasetUtil(mnist_dataset)
    # transform dataset to tfrecords
    dataset.to_tfrecords(output_file_path)

    # ======== Load tfrecord ========
    dataset = DatasetUtil(shape=[28, 28, 1])
    dataset = dataset.from_tfrecords(output_file_path)

    image, label = next(iter(dataset))
    plt.title(str(label))
    plt.imshow(image.numpy())
    plt.show()
