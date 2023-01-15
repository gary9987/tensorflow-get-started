import numpy as np
import os
from typing import List


def calculate_mean_std_by_order(all_files: List, idx: int, order: int):
    target = np.array([], dtype=float)

    for file in all_files:
        if 'npz' in file:
            data = np.load(file)
            x = data['x']
            target = np.append(target, x[:, idx]**order)

    return np.mean(target), np.std(target)


if __name__ == '__main__':
    # file_path = 'LearningCurveDataset'
    file_path = 'NasBench101Dataset'
    folders = os.listdir(file_path)

    all_files = []
    for folder in folders:
        for file in os.listdir(os.path.join(file_path, folder)):
            all_files.append(os.path.join(file_path, folder, file))

    for order in range(1, 7):
        print(f'flops order{order}: ', calculate_mean_std_by_order(all_files, 7, order))

    for order in range(1, 7):
        print(f'params order{order}: ', calculate_mean_std_by_order(all_files, 8, order))
    """
    flops order1: (26638819.405752193, 65364232.74239259)
    flops order2:  (4982109621333941.0, 1.9215401177096412e+16)
    flops order3:  (1.3407101694764562e+24, 5.818062132947369e+24)
    flops order4:  (3.940530586757471e+32, 1.759003441563639e+33)
    flops order5:  (1.1822919313908888e+41, 5.313650616742194e+41)
    flops order6:  (3.5647350741373663e+49, 1.6047937641856067e+50)
    params order1:  (90818.1944483277, 316351.1693478308)
    params order2:  (108326006790.59427, 693711124748.4598)
    params order3:  (2.1707989633206323e+17, 1.6338307382838333e+18)
    params order4:  (4.92969648346968e+23, 3.859175310803615e+24)
    params order5:  (1.1531006080581303e+30, 9.114026410774042e+30)
    params order6:  (2.716526562752633e+36, 2.152212202322697e+37)    
    """

    '''
    count = 0
    flops = np.array([], dtype=float)
    params = np.array([], dtype=float)

    for file in all_files:
        if 'npz' in file:
            data = np.load(file)
            x = data['x']
            count += x.shape[0]
            flops = np.append(flops, x[:, 9])
            params = np.append(params, x[:, 9])

    print('flops mean = {}, std = {}'.format(str(np.mean(flops)), str(np.std(flops))))

    print('params mean = {}, std = {}'.format(str(np.mean(params)), str(np.std(params))))
    '''

    """
    learning_curve_dataset
    flops mean = 28819043.233719405, std = 68531284.19735347
    params mean = 98277.40047462686, std = 332440.6417713961
    """

    """
    nas_bench_101_dataset
    flops mean = 28108567.14472483, std = 67398823.71203184
    params mean = 95841.84206601226, std = 326745.84386388084
    """

    """
    nas_bench_101_dataset feature9: layer 
    flops mean = 30.528941727204867, std = 17.807043336964252
    """