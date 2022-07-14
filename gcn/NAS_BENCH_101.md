# Nas-Bench-101 usage
## Setup
- Download the full record
    ```bash
    cd tensorflow-get-started/gcn
    mkdir nas-bech-101-data
    cd nas-bech-101-data
    curl -O https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
    ```
- Clone Nas-Bench-101 api and use the patch file to modify
    ```bash
    cd tensorflow-get-started/gcn
    git clone https://github.com/google-research/nasbench
    patch -p1 -i api.patch 
    pip3 install ./nasbench
    ```
## Ops fromat
```python
INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
```
## Usage
```python
from nasbench import api

nasbench = api.NASBench('./nas-bench-101-data/nasbench_only108.tfrecord')

# Query an Inception-like cell from the dataset.
cell = api.ModelSpec(
  matrix=[[0, 1, 1, 1, 0, 1, 0],    # input layer
          [0, 0, 0, 0, 0, 0, 1],    # 1x1 conv
          [0, 0, 0, 0, 0, 0, 1],    # 3x3 conv
          [0, 0, 0, 0, 1, 0, 0],    # 5x5 conv (replaced by two 3x3's)
          [0, 0, 0, 0, 0, 0, 1],    # 5x5 conv (replaced by two 3x3's)
          [0, 0, 0, 0, 0, 0, 1],    # 3x3 max-pool
          [0, 0, 0, 0, 0, 0, 0]],   # output layer
  # Operations at the vertices of the module, matches order of matrix.
  ops=[INPUT, CONV1X1, CONV3X3, CONV3X3, CONV3X3, MAXPOOL3X3, OUTPUT])

# Querying multiple times may yield different results. Each cell is evaluated 3
# times at each epoch budget and querying will sample one randomly.
data = nasbench.query(cell, query_idx = 0)
for k, v in data.items():
  print('%s: %s' % (k, str(v)))
  
'''
module_adjacency: [[0 1 1 1 1 0 0]
 [0 0 0 0 0 0 1]
 [0 0 0 0 0 0 1]
 [0 0 0 0 0 0 1]
 [0 0 0 0 0 1 0]
 [0 0 0 0 0 0 1]
 [0 0 0 0 0 0 0]]
module_operations: ['input', 'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3', 'conv3x3-bn-relu', 'conv3x3-bn-relu', 'output']
trainable_parameters: 2694282
training_time: 1155.85302734375
train_accuracy: 1.0
validation_accuracy: 0.9376001358032227
test_accuracy: 0.9311898946762085
'''
```
# Nas-Bench-101-Dataset
- The channels of feature x means
  ```python
  self.features_dict = {'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4,
                        'Classifier': 5, 'maxpool2x2': 6, 'flops': 7, 'params': 8, 'num_layer': 9,
                        'input_shape_1': 10, 'input_shape_2': 11, 'input_shape_3': 12, 'output_shape_1': 13,
                        'output_shape_2': 14, 'output_shape_3': 15}
  ```
- The label y format
  ```python
  # Labels Y
  y = np.zeros((3, 4))  # 3 x (train_accuracy, validation_accuracy, test_accuracy, training_time)
  for query_idx in range(3):
      y[query_idx][0] = record[2][query_idx]['train_accuracy']
      y[query_idx][1] = record[2][query_idx]['validation_accuracy']
      y[query_idx][2] = record[2][query_idx]['test_accuracy']
      y[query_idx][3] = record[2][query_idx]['training_time']
  ```