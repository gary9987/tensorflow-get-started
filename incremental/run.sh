# cifar10 (None, 32, 32, 3)
# cifar100 (None, 32, 32, 3)
# mnist (None, 28, 28, 1)
python3 incremental_training.py --start 0 --end 249 --dataset_name 'cifar10' --inputs_shape '(-1, 32, 32, 3)'
