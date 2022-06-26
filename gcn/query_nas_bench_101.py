# nasbench should be running on tf1.x
import pickle
from nasbench import api

if __name__ == '__main__':
    file = open('../incremental/cell_list.pkl', 'rb')
    cell_list = pickle.load(file)
    file.close()

    nasbench = api.NASBench('./nas-bench-101-data/nasbench_full.tfrecord')

    new_list = []
    
    print('cell_list length = {}'.format(str(len(cell_list))))
    
    count = 0
    
    for cell in cell_list:
        
        if count % 10000 == 0:
            print('now processing {}'.format(str(count)))
       
        count += 1
        matrix = cell[0]
        try:
            ops = list(map(lambda x: x.lower(), cell[1]))
            data = nasbench.query(api.ModelSpec(matrix=matrix, ops=ops))
            new_list.append([matrix, cell[1], data])
        except:
            print('skip with error ', matrix, cell[1])

    with open('./nas-bench-101-data/nasbench_101_cell_list.pkl', 'wb') as f:
        pickle.dump(new_list, f)
    
        






