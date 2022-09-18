import numpy as np
import os

if __name__ == '__main__':

    file_path = 'Preprocessed_NasBench101Dataset'

    out = [np.zeros((1, 3))] * 3

    for no in range(0, 169594):#169594):
        data = np.load(os.path.join(file_path, f'graph_{no}.npz'))
        y = data['y']

        row = []
        for i in range(3):
            """
            train: 0
            valid: 1
            test:  2
            """
            row.append([y[0][i] - y[1][i], y[1][i] - y[2][i], y[2][i] - y[0][i]])

            row[i] = np.array(row[i]).reshape(1, 3)
            out[i] = np.append(out[i], row[i], axis=0)

    for i, j in enumerate(['train', 'valid', 'test']):
        out[i] = np.delete(out[i], 0, axis=0)
        np.savez(f'out_{j}.npz', out=out[i])

    for i in ['train', 'valid', 'test']:
        data = np.load(f'out_{i}.npz')['out']
        print(data.shape)
    # (169594, 3)


