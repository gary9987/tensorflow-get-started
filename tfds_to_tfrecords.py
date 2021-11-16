import tensorflow as tf
import os
import tensorflow_datasets as tfds


if __name__ == '__main__':

    mnist_dataset, metadata = tfds.load(
        'mnist',
        split='train',
        with_info=True,
        as_supervised=True,
        batch_size=1
    )
    mnist_dataset = mnist_dataset.prefetch(tf.data.AUTOTUNE)

    output_file_path = './output/'
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    tfrecord_file = output_file_path + 'train.tfrecords'

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for image, label in mnist_dataset:
            image = image.numpy().tobytes()
            label = int(label)
            feature = {  # create tf.train.Feature dictionary
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # Image is a byte list
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # Label is a int64
            }
            record_bytes = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(record_bytes.SerializeToString())



