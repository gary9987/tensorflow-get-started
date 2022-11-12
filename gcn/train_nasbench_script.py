import subprocess

if __name__ == '__main__':
    for i in range(10, 91, 10):
        print(f'Now running python3 train_nasbench.py --mid_point {i} --model_output_dir full_model')
        subprocess.check_output(f'python3 train_nasbench.py --mid_point {i} --model_output_dir full_model', shell=True)