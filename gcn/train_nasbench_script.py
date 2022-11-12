import subprocess

if __name__ == '__main__':
    for i in range(10, 91, 10):
        cmd = f'python3 train_nasbench.py --mid_point {i} --model_output_dir full_model'
        print(f'Now running {cmd}')
        subprocess.check_output(cmd, shell=True)