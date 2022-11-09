import numpy as np
import os

if __name__ == '__main__':
    out = [np.zeros((1, 3))] * 3

    more_than_80 = 0
    under80 = 0

    for size in range(3, 7+1):

        #file_path = f'Preprocessed_NasBench101Dataset/Preprocessed_NasBench101Dataset_{size}'
        file_path = f'NasBench101Dataset/NasBench101Dataset_{size}'

        for no in range(0, 169593+1):  #169593+1):
            try:
                data = np.load(os.path.join(file_path, f'graph_{no}.npz'))
                y = data['y']

                for idx in range(1, 2):
                    if not np.isnan(y[idx][1]):
                        if y[idx][1] <= 0.8:
                            under80 += 1
                        else:
                            more_than_80 += 1
                        break
            except:
                break

    print(more_than_80, under80)
    # All Filtered        167956 722
    # train 0~120000      118852 498
    # valid 120001~145000: 24768, 98
    # test  145001~169593: 24336, 126

    # All Filtered all size 193799 818

    # All Non-filtered    168012 1582
    # train 0~120000      118889  1112
    # valid 120001~145000: 24776, 224
    # test  145001~169593: 24347, 246

    # All Non-filtered all size 193856 1843


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

