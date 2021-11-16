# tensorflow-get-started
A learning note.
## tensorflow_dataset
### tfds
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
### Data transformation
- dataset.map: using gpu acceleration to do transform.
### Prefetch
- dataset = dataset.prefetch(buffer_size)
  - If the value tf.data.AUTOTUNE is used, then the buffer size is dynamically tuned.
### Cache
- dataset = dataset.cache(filename='')
    - If a filename is not provided, the dataset will be cached in memory.
## TFRecords
A tensorflow dataset format.
- [Reference](https://tf.wiki/zh_hans/basic/tools.html#tfrecord)
