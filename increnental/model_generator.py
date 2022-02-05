from model import ResBlock, CustomBranch
import tensorflow as tf

'''
def CustomInceptionModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(CustomBranch())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(2, 2), padding='same'))
    return model
'''

def model_generator(para = None):
    """
    Conv2D: ['Conv2D filter kernel_x kernel_y padding stride_x stride_y']
    MaxPooling2D: ['MaxPooling2D pool_x pool_y padding stride_x stride_y']
    AveragePooling2D: ['AveragePooling2D pool_x pool_y padding stride_x stride_y']
    ResBlock: ['ResBlock filter stride_x stride_y']
    Activation: ['Activation (name)'] ((name) can be 'relu', 'sigmoid', 'tanh', 'softmax', ...])
    BatchNormalization: ['BatchNormalization axis']
    """
    if para is None:
        branch_par = [['Conv2D 64 3 3 same 1 1', 'ResBlock 64 1 1'],
                      ['Conv2D 96 1 1 same 1 1', 'Conv2D 128 3 3 same 1 1'],
                      ['Conv2D 16 1 1 same 1 1', 'Conv2D 32 5 5 same 1 1'],
                      ['AveragePooling2D 3 3 same 1 1', 'Conv2D 32 1 1 same 1 1']]
        para = ['Conv2D 192 3 3 same 1 1', 'BatchNormalization 3', 'MaxPooling2D 3 3 same 2 2', branch_par,
                'Activation relu', 'AveragePooling2D 7 7 same 2 2']

    # Init the Sequential Model
    model = tf.keras.Sequential()

    # Add layer by layer
    for layer in para:
        # If meet the Branch layer
        if type(layer) == list:
            model.add(CustomBranch(branch_par=layer))
        else:
            layers = layer.split()
            if layers[0] == 'Conv2D':
                filters = int(layers[1])
                kernel = (int(layers[2]), int(layers[3]))
                padding = layers[4]
                stride = (int(layers[5]), int(layers[6]))
                model.add(tf.keras.layers.Conv2D(filters, kernel, padding=padding, strides=stride, name=None))
            elif layers[0] == 'MaxPooling2D':
                pool_size = (int(layers[1]), int(layers[2]))
                padding = layers[3]
                stride = (int(layers[4]), int(layers[5]))
                model.add(tf.keras.layers.MaxPooling2D(pool_size, strides=stride, padding=padding))
            elif layers[0] == 'AveragePooling2D':
                pool_size = (int(layers[1]), int(layers[2]))
                padding = layers[3]
                stride = (int(layers[4]), int(layers[5]))
                model.add(tf.keras.layers.AveragePooling2D(pool_size, strides=stride, padding=padding))
            elif layers[0] == 'ResBlock':
                filters = int(layers[1])
                stride = (int(layers[2]), int(layers[3]))
                model.add(ResBlock(filters, strides=stride))
            elif layers[0] == 'Activation':
                act = layers[1]
                model.add(tf.keras.layers.Activation(act))
            elif layers[0] == 'BatchNormalization':
                axis = int(layers[1])
                model.add(tf.keras.layers.BatchNormalization(axis=axis))
            else:
                print('Error, the layer ', layers[0], 'type not defined.')
                exit()

    return model
