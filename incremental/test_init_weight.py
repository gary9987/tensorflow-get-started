import tensorflow as tf
from model import Classifier
import model_util

seed = 0

if __name__ == '__main__':
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.1, momentum=0.9, epsilon=1.0)

    tf.random.set_seed(seed)
    model = model_util.get_model_by_id_and_layer('./cell_list.pkl', seed, (None, 28, 28, 1), 0, 10)
    model.add(Classifier(10))

    tf.random.set_seed(seed)
    # Create a new model and copy the weight
    model2 = model_util.get_model_by_id_and_layer('./cell_list.pkl', seed, (None, 28, 28, 1), 0, 10)
    model2.add(Classifier(10))

    for i in range(len(model.layers)):
        a = model.layers[i].get_weights()
        b = model2.layers[i].get_weights()
        print('Check', i, 'layer...')

        for j in range(len(a)):
            check_all = (a[j] == b[j]).reshape(-1).all()
            if not check_all:
                print('Not equal.')




    