import numpy as np
import os

if __name__ == '__main__':

    #file_path = 'Preprocessed_NasBench101Dataset'
    file_path = 'NasBench101Dataset'

    out = [np.zeros((1, 3))] * 3

    more_than_80 = 0
    under80 = 0

    for no in range(0, 169594):  #169594):
        data = np.load(os.path.join(file_path, f'graph_{no}.npz'))
        y = data['y']

        for idx in range(y.shape[0]):
            if not np.isnan(y[idx][0]):
                if y[idx][0] < 0.8:
                    under80 += 1
                else:
                    more_than_80 += 1
                break

    print(more_than_80, under80)
    # Filtered 169042 387
    # Non-filtered 168403 1191


    '''
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
        '''

    '''
    for i, j in enumerate(['train', 'valid', 'test']):
        out[i] = np.delete(out[i], 0, axis=0)
        np.savez(f'out_{j}.npz', out=out[i])

    for i in ['train', 'valid', 'test']:
        data = np.load(f'out_{i}.npz')['out']
        print(data.shape)
    # (169594, 3)
    '''
