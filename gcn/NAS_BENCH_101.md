# Nas-Bench-101 usage
## Setup
- Download the full record
    ```bash
    mkdir nas-bech-101-data
    cd nas-bech-101-data
    curl -O https://storage.googleapis.com/nasbench/nasbench_full.tfrecord
    ```
- Clone Nas-Bench-101 api
    ```bash
    git clone https://github.com/google-research/nasbench
    ```
- Modify `./nasbench/api.py`
  - Replace the `query()` function
  ```python
    def query(self, model_spec, query_idx, epochs=108, stop_halfway=False):
      """Fetch one of the evaluations for this model spec.
  
      Each call will sample one of the config['num_repeats'] evaluations of the
      model. This means that repeated queries of the same model (or isomorphic
      models) may return identical metrics.
  
      This function will increment the budget counters for benchmarking purposes.
      See self.training_time_spent, and self.total_epochs_spent.
  
      This function also allows querying the evaluation metrics at the halfway
      point of training using stop_halfway. Using this option will increment the
      budget counters only up to the halfway point.
  
      Args:
        model_spec: ModelSpec object.
        epochs: number of epochs trained. Must be one of the evaluated number of
          epochs, [4, 12, 36, 108] for the full dataset.
        stop_halfway: if True, returned dict will only contain the training time
          and accuracies at the halfway point of training (num_epochs/2).
          Otherwise, returns the time and accuracies at the end of training
          (num_epochs).
  
      Returns:
        dict containing the evaluated data for this object.
  
      Raises:
        OutOfDomainError: if model_spec or num_epochs is outside the search space.
      """
      if epochs not in self.valid_epochs:
        raise OutOfDomainError('invalid number of epochs, must be one of %s'
                               % self.valid_epochs)
  
      fixed_stat, computed_stat = self.get_metrics_from_spec(model_spec)
      # sampled_index = random.randint(0, self.config['num_repeats'] - 1)
      computed_stat = computed_stat[epochs][query_idx]
  
      data = {}
      data['module_adjacency'] = fixed_stat['module_adjacency']
      data['module_operations'] = fixed_stat['module_operations']
      data['trainable_parameters'] = fixed_stat['trainable_parameters']
  
      if stop_halfway:
        data['training_time'] = computed_stat['halfway_training_time']
        data['train_accuracy'] = computed_stat['halfway_train_accuracy']
        data['validation_accuracy'] = computed_stat['halfway_validation_accuracy']
        data['test_accuracy'] = computed_stat['halfway_test_accuracy']
      else:
        data['training_time'] = computed_stat['final_training_time']
        data['train_accuracy'] = computed_stat['final_train_accuracy']
        data['validation_accuracy'] = computed_stat['final_validation_accuracy']
        data['test_accuracy'] = computed_stat['final_test_accuracy']
  
      self.training_time_spent += data['training_time']
      if stop_halfway:
        self.total_epochs_spent += epochs // 2
      else:
        self.total_epochs_spent += epochs
  
      return data
  ```
- Install
  ```bash
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