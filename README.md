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
### Write a record to a file
```python
output_file_path = './output/'
tfrecord_file = output_file_path + 'train.tfrecords'
with tf.io.TFRecordWriter(tfrecord_file) as writer:
    for image, label in mnist_dataset:
        image = image.numpy().tobytes() # transform dataset nd.array to a byte list
        label = int(label) # transform dataset label nd.array to a int64
        feature = {  # create tf.train.Feature dictionary
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),  # Image is a byte list
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))  # Label is a int64
        }
        record_bytes = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(record_bytes.SerializeToString())
```