# nasbench should be running on tf1.x
import os
import pickle
import subprocess
from pathlib import Path
from nasbench import api
import numpy as np

if __name__ == '__main__':
    file = open('../incremental/cell_list.pkl', 'rb')
    cell_list = pickle.load(file)
    file.close()

    nasbench_data_dir = Path('nas-bench-101-data')
    nasbench_only108_filename = Path('nasbench_only108.tfrecord')
    if not os.path.exists(nasbench_data_dir / nasbench_only108_filename):
        os.makedirs(nasbench_data_dir)
        subprocess.check_output(
            f'wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord; mv {nasbench_only108_filename.name} {nasbench_data_dir}',
            shell=True)

    nasbench = api.NASBench(str(nasbench_data_dir / nasbench_only108_filename))

    new_list = []
    
    print('cell_list length = {}'.format(str(len(cell_list))))
    
    count = 0
    skip_count = 0

    for cell in cell_list:
    #for cell in cell_list[:10]:    
        if count % 10000 == 0:
            print('now processing {}'.format(str(count)))
       
        count += 1
        matrix = cell[0]
        try:
            ops = list(map(lambda x: x.lower(), cell[1]))
            data = [nasbench.query(api.ModelSpec(matrix=matrix, ops=ops), query_idx = i) for i in range(3)]
            #print(data)
            new_list.append([matrix, cell[1], data])
        except:
            skip_count += 1
            #print('skip with error ', matrix, cell[1])

    with open(nasbench_data_dir / Path('nasbench_101_cell_list.pkl'), 'wb') as f:
        pickle.dump(new_list, f)
    
    print(f'Finish with skip {skip_count}')
        






