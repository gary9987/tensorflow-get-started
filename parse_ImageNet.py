import tensorflow as tf
import numpy as np

def parse_example_proto(example_serialized):
    """Parses an Example proto containing a training example of an image.

    The output of the build_image_data.py image preprocessing script is a dataset
    containing serialized Example protocol buffers. Each Example proto contains
    the following fields:

        image/height: 462
        image/width: 581
        image/colorspace: 'RGB'
        image/channels: 3
        image/class/label: 615
        image/class/synset: 'n03623198'
        image/class/text: 'knee pad'
        image/object/bbox/xmin: 0.1
        image/object/bbox/xmax: 0.9
        image/object/bbox/ymin: 0.2
        image/object/bbox/ymax: 0.6
        image/object/bbox/label: 615
        image/format: 'JPEG'
        image/filename: 'ILSVRC2012_val_00041207.JPEG'
        image/encoded: <JPEG encoded string>

    Args:
        example_serialized: scalar Tensor tf.string containing a serialized
            Example protocol buffer.

    Returns:
        image_buffer: Tensor tf.string containing the contents of a JPEG file.
        label: Tensor tf.int32 containing the label.
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged as
            [ymin, xmin, ymax, xmax].
        text: Tensor tf.string containing the human-readable label.
    """
    # Dense features in Example proto.
    feature_map = {
            'image/encoded': tf.compat.v1.FixedLenFeature([], dtype=tf.string,
                                                                                    default_value=''),
            'image/class/label': tf.compat.v1.FixedLenFeature([1], dtype=tf.int64,
                                                                                            default_value=-1),
            'image/class/text': tf.compat.v1.FixedLenFeature([], dtype=tf.string,
                                                                                         default_value=''),
    }
    sparse_float32 = tf.compat.v1.VarLenFeature(dtype=tf.float32)
    # Sparse features in Example proto.
    feature_map.update(
            {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                                                     'image/object/bbox/ymin',
                                                                     'image/object/bbox/xmax',
                                                                     'image/object/bbox/ymax']})

    features = tf.compat.v1.parse_single_example(example_serialized, feature_map)
    label = tf.compat.v1.cast(features['image/class/label'], dtype=tf.int32)

    xmin = tf.compat.v1.expand_dims(features['image/object/bbox/xmin'].values, 0)
    ymin = tf.compat.v1.expand_dims(features['image/object/bbox/ymin'].values, 0)
    xmax = tf.compat.v1.expand_dims(features['image/object/bbox/xmax'].values, 0)
    ymax = tf.compat.v1.expand_dims(features['image/object/bbox/ymax'].values, 0)
    #print(xmax, xmin, ymax, ymin)
    # Note that we impose an ordering of (y, x) just to make life difficult.
    bbox = tf.compat.v1.concat([ymin, xmin, ymax, xmax], 0)
    #print(bbox)
    # Force the variable number of bounding boxes into the shape
    # [1, num_boxes, coords].
    bbox = tf.expand_dims(bbox, 0)
    bbox = tf.transpose(bbox, [0, 2, 1])

    return features['image/encoded'], label, bbox, features['image/class/text']


if __name__ == '__main__':

    filename = '/home/gary/Documents/processed_data/train-00000-of-01024'
    raw_dataset = tf.data.TFRecordDataset(filename)
    raw_dataset = raw_dataset.map(parse_example_proto)
    for a, b, c, d in raw_dataset:
        print(c)
