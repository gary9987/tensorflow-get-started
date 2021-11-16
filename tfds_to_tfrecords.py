import tensorflow as tf
import os
import tensorflow_datasets as tfds


class TfdsToTFRecords:
    def __init__(self, dataset):
        """
        :param dataset: give a tfds dataset with supervised
        """
        self.dataset = dataset.prefetch(tf.data.AUTOTUNE)
        self.dataset = self.dataset.batch(1)

    def to_tfrecords(self, filename='output.tfrecords'):
        """
        :param filename: give a output file path
        """

        if not os.path.exists(os.path.dirname(filename)):
            if os.path.dirname(filename) != '':
                os.makedirs(os.path.dirname(filename))

        with tf.io.TFRecordWriter(filename) as writer:
            for image, label in mnist_dataset:
                image = image.numpy().tobytes()
                label = int(label)
                feature = {  # create tf.train.Feature dictionary
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # Image is a byte list
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # Label is a int64
                }
                record_bytes = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(record_bytes.SerializeToString())


if __name__ == '__main__':
    """
    The example of TfdsToTFRecords usage.
    """
    mnist_dataset, metadata = tfds.load(
        'mnist',
        split='train',
        with_info=True,
        as_supervised=True,
    )

    output_file_path = './output/train.tfrecords'
    helper = TfdsToTFRecords(mnist_dataset)
    helper.to_tfrecords(output_file_path)
