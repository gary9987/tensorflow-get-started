# tensorflow-get-started
A learning note.
## tensorflow_dataset
### import tensorflow_dataset as tfds
```python=
# load tf_flowers dataset
dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)

# using map to do data augmentation
dataset = dataset.map(lambda img, label: (tf.image.resize(img, [224, 224]) / 255.0, label)).shuffle(1024).batch(32)

# let CPU prefecting data
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for images, labels in dataset:
    # training...
```
- dataset.map: using gpu acceleration to do transform

## TFRecords
A tensorflow dataset format.
- [Reference](https://tf.wiki/zh_hans/basic/tools.html#tfrecord)
