# Incremental Training
## Environment
- Python 3.8.10
- Tensorflow 2.7.0
- Others in `requirements.txt`
## Steps
### 1. Download the `cell_list.pkl`
```bash
sh download.sh
```
### 2. Run `incremental_training.py`
```bash
python3 incremental_training.py
```
## Working on CML server
### 1. Setup conda environment
```bash
conda create --name ${env_name} python=3.8
conda activate ${env_name}
pip3 install -r requirements.txt
```
### 2. Set environment variable
- Content in `load_cuda.sh`
    ```bash
    module load cuda/11.2
    module load cudnn/cuda112_8.1
    export CUDA_VISIBLE_DEVICES=0
    export NUMEXPR_MAX_THREADS=16
    ```
- Set `CUDA_VISIBLE_DEVICES` to non-working gpu by edit the `load_cuda.sh`.
- load
    ```bash
    source load_cuda.sh
    ```
### 3. Download the `cell_list.pkl`
```bash
cd incremental
sh download.sh
```
### 4. Run `incremental_training.py`
```bash
python3 incremental_training.py
```