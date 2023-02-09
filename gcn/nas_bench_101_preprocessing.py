import pickle
import numpy as np
import os

if __name__ == '__main__':
    for size in range(3, 8):
        cell_list_filename = f'nas-bench-101-data/nasbench_101_cell_list_{size}.pkl'
        with open(cell_list_filename, 'rb') as f:
            cell_list = pickle.load(f)

        in_path = f'NasBench101Dataset/NasBench101Dataset_{size}'
        out_path = f'Preprocessed_NasBench101Dataset/Preprocessed_NasBench101Dataset_{size}'

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        for no in range(0, len(cell_list)):  # 169594
            data = np.load(os.path.join(in_path, f'graph_{no}.npz'))
            y = data['y']

            row = []
            for i in range(3):
                """
                train: 0
                valid: 1
                test:  2
                """

                # [value, original index]
                row.append([y[i][1], i])

            row.sort(reverse=True)
            new_y = np.full((3, 4), np.nan)

            if row[0][0] - row[1][0] < 0.1 and row[1][0] - row[2][0] < 0.1:
                for i in range(3):
                    new_y[row[i][1]] = y[row[i][1]]
            elif row[0][0] - row[1][0] < 0.1 and row[2][0] < 0.2:
                for i in range(2):
                    new_y[row[i][1]] = y[row[i][1]]
            elif row[0][0] - row[1][0] > 0.1 and row[2][0] < 0.2:
                for i in range(1):
                    new_y[row[i][1]] = y[row[i][1]]
            elif row[0][0] > 0.3 and row[1][0] < 0.2 and row[2][0] < 0.2:
                for i in range(1):
                    new_y[row[i][1]] = y[row[i][1]]
            elif 0.3 > row[0][0] > 0.2 and row[1][0] < 0.2 and row[2][0] < 0.2:
                if row[1][0] - row[2][0] > 0.1:
                    for i in range(2):
                        new_y[row[i][1]] = y[row[i][1]]
                else:
                    for i in range(3):
                        new_y[row[i][1]] = y[row[i][1]]
            elif row[0][0] < 0.2 and row[1][0] < 0.2 and row[2][0] < 0.2:
                for i in range(3):
                    new_y[row[i][1]] = y[row[i][1]]

            np.savez(os.path.join(out_path, f'graph_{no}.npz'), a=data['a'], x=data['x'], e=data['e'], y=new_y)

