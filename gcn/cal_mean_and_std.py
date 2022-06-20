import numpy as np
import os


if __name__ == '__main__':
    file_path = 'LearningCurveDataset'
    files = os.listdir(file_path)

    count = 0
    flops = np.array([], dtype=float)
    params = np.array([], dtype=float)

    for i in files:
        if 'npz' in i:
            data = np.load(os.path.join(file_path, i))
            x = data['x']
            count += x.shape[0]
            flops = np.append(flops, x[:, 7])
            params = np.append(params, x[:, 8])

    print('flops mean = {}, std = {}'.format(str(np.mean(flops)), str(np.std(flops))))

    print('params mean = {}, std = {}'.format(str(np.mean(params)), str(np.std(params))))

    """
    flops mean = 28819043.233719405, std = 68531284.19735347
    params mean = 98277.40047462686, std = 332440.6417713961
    """
