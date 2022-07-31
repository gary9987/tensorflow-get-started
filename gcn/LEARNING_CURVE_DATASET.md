# Learning-Curve-Dataset
- The channels of feature x means
  ```python
  self.features_dict = {'INPUT': 0, 'conv1x1-bn-relu': 1, 'conv3x3-bn-relu': 2, 'maxpool3x3': 3, 'OUTPUT': 4,
                              'Classifier': 5, 'maxpool2x2': 6, 'flops': 7, 'params': 8, 'num_layer': 9}
- The label y format
  ```python
  # Labels Y
  y = np.zeros(3)  # train_acc, valid_acc, test_acc
  with open(self.record_dir + log_file) as f:   
      rows = csv.DictReader(f)
      tmp = []
      for row in rows:
          tmp.append(row)
          y[0] = tmp[-1]['accuracy']
          y[1] = tmp[-1]['val_accuracy']
          y[2] = tmp[-1]['test_acc']
  ```