# Combine
## Steps
### 1. Move `combine.py` to directory which contains the file to be combined
- The files in directory would be:
    ```bash
    .
    ├── cifar10_log_1.tar
    ├── cifar10_log_2.tar
    └── combine.py
    ```
### 2. Run `combine.py` with dataset name as argument.
```bash
python3 combine.py --name ${dataset_name}
```
- For example
```bash
python3 combine.py --name cifar10
```
### 3. All the files will be merged into a directory named `combine`.
- The log will be:
    ```log
    INFO:root:There are [PosixPath('cifar10_log_2.tar'), PosixPath('cifar10_log_1.tar')] to be proccessed.
    INFO:root:Copy 15384 files to combined.
    INFO:root:There are [PosixPath('cifar10_log_1/cifar10.pkl'), PosixPath('cifar10_log/cifar10.pkl')] to be merged.
    INFO:root:The length of merged cifar10.pkl is 15384.
    INFO:root:Pickle dump to ./combined/cifar10.pkl. complete.
    ```
