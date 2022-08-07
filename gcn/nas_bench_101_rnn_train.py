import logging
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers

logging.basicConfig(filename='nas_bench_101_dataset.log', level=logging.INFO)


def bfs(matrix):
    todo = [1]
    visit = [False] * 8
    visit[1] = True
    order = []

    num_nodes = 7
    while len(todo) != 0:
        r = len(todo)
        for i in range(r):
            front = todo.pop(0)
            order.append(front)
            for j in range(1, num_nodes + 1):
                if matrix[front][j] != 0 and visit[j] is not True:
                    todo.append(j)
                    visit[j] = True

    return order


def get_dataset(start, end, label_query):
    file_path = 'NasBench101Dataset'
    x_train = []
    y_train = []

    for no in range(start, end + 1):
        data = np.load(os.path.join(file_path, f'graph_{no}.npz'))
        x = data['x']
        y = data['y']
        a = data['a']

        y = y[label_query][:3]

        order = bfs(a)
        out = np.zeros((a.shape[0], x.shape[1]), dtype=float)

        now_idx = 0
        for now_layer in range(12):
            if now_layer == 0:
                out[now_idx] = x[0]
                now_idx += 1
            elif now_layer == 4:
                out[now_idx] = x[22]
                now_idx += 1
            elif now_layer == 8:
                out[now_idx] = x[44]
                now_idx += 1
            else:
                now_group = now_layer // 4 + 1
                node_start_no = now_group + 7 * (now_layer - now_group)

                for i in range(len(order)):
                    real_idx = node_start_no + order[i] - 1
                    out[now_idx] = x[real_idx]
                    now_idx += 1

        x_train.append(out)
        y_train.append(y)

    x_train, y_train = np.array(x_train), np.array(y_train)
    return x_train, y_train


if __name__ == '__main__':
    x_train, y_train = get_dataset(start=0, end=80000, label_query=0)
    x_valid, y_valid = get_dataset(start=80001, end=160000, label_query=1)
    x_test, y_test = get_dataset(80001, 80256, 0)

    model = tf.keras.Sequential()
    model.add(layers.LSTM(128, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(3))
    print(model.summary())
    model.compile('adam', 'mean_squared_error', metrics=['mse'])
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=100, batch_size=64)

    pred = model.predict(x_test)
    for i, j in zip(y_test, pred):
        print(i, j)


