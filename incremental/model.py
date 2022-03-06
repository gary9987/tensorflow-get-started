import keras
import tensorflow as tf
from tensorflow.keras import layers


class ResBlock(layers.Layer):
    """
    If the stride is not equal to 1 or the filters of the input is not equal to given filter_nums, then it will need a
    Con1x1 layer with given stride to project the input.
    """
    def __init__(self, filters, strides=(1, 1)):
        super(ResBlock, self).__init__()
        self.filters = filters
        self.strides = strides
        self.is_con1x1_build = False

        self.conv_1 = layers.Conv2D(filters, (3, 3), strides=strides, padding='same')
        self.bn_1 = layers.BatchNormalization()
        self.act_relu = layers.Activation('relu')

        self.conv_2 = layers.Conv2D(filters, (3, 3), strides=1, padding='same')
        self.bn_2 = layers.BatchNormalization()

        self.identity_block = lambda x: x

    def call(self, inputs):

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = self.act_relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        """
        The identity_block will change to Conv1x1 while the stride != 1 or input filter != given filter_nums.
        """
        if inputs.shape[3] != self.filters or self.strides != (1, 1):
            # If the identity block haven't been changed.
            if not self.is_con1x1_build:
                self.identity_block = tf.keras.Sequential()
                self.identity_block.add(layers.Conv2D(self.filters, (1, 1), strides=self.strides))
                self.is_con1x1_build = True

        short_cut = self.identity_block(inputs)

        outputs = layers.add([x, short_cut])
        outputs = tf.nn.relu(outputs)
        return outputs


class CustomBranch(tf.keras.Model):
    """
    Conv2D: ['Conv2D filter kernel_x kernel_y padding stride_x stride_y']
    MaxPooling2D: ['MaxPooling2D pool_x pool_y padding stride_x stride_y']
    AveragePooling2D: ['AveragePooling2D pool_x pool_y padding stride_x stride_y']
    ResBlock: ['ResBlock filter stride_x stride_y']
    Activation: ['Activation (name)'] ((name) can be 'relu', 'sigmoid', 'tanh', 'softmax', ...])
    """
    def __init__(self, branch_par=None):
        super(CustomBranch, self).__init__()

        if branch_par is None:
            branch_par = [['Conv2D 64 3 3 same 1 1', 'ResBlock 64 1 1'], ['Conv2D 96 1 1 same 1 1', 'Conv2D 128 3 3 same 1 1'],
                          ['Conv2D 16 1 1 same 1 1', 'Conv2D 32 5 5 same 1 1'], ['AveragePooling2D 3 3 same 1 1', 'Conv2D 32 1 1 same 1 1']]

        self.branch_list = []

        for branch in branch_par:

            a_branch = []

            for layer in branch:
                layers = layer.split()
                if layers[0] == 'Conv2D':
                    filters = int(layers[1])
                    kernel = (int(layers[2]), int(layers[3]))
                    padding = layers[4]
                    stride = (int(layers[5]), int(layers[6]))
                    a_branch.append(tf.keras.layers.Conv2D(filters, kernel, padding=padding, strides=stride, name=None))
                elif layers[0] == 'MaxPooling2D':
                    pool_size = (int(layers[1]), int(layers[2]))
                    padding = layers[3]
                    stride = (int(layers[4]), int(layers[5]))
                    a_branch.append(tf.keras.layers.MaxPooling2D(pool_size, strides=stride, padding=padding))
                elif layers[0] == 'AveragePooling2D':
                    pool_size = (int(layers[1]), int(layers[2]))
                    padding = layers[3]
                    stride = (int(layers[4]), int(layers[5]))
                    a_branch.append(tf.keras.layers.AveragePooling2D(pool_size, strides=stride, padding=padding))
                elif layers[0] == 'ResBlock':
                    filters = int(layers[1])
                    stride = (int(layers[2]), int(layers[3]))
                    a_branch.append(ResBlock(filters, strides=stride))
                elif layers[0] == 'Activation':
                    act = layers[1]
                    a_branch.append(tf.keras.layers.Activation(act))
                else:
                    print('Error, the layer ', layers[0], 'type not defined.')
                    exit()

            self.branch_list.append(a_branch)


        '''
        self.branch1x1 = tf.keras.layers.Conv2D(branch1[0], (1, 1), padding='same', strides=(1, 1), name=None)
        self.branchpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branchpool_2 = tf.keras.layers.Conv2D(branch4[0], (1, 1), padding='same', strides=(1, 1), name=None)
        '''

    def call(self, inputs):
        outputs = []
        for branch in self.branch_list:
            x = 0
            for i in range(len(branch)):
                if i == 0:
                    x = branch[i](inputs)
                else:
                    x = branch[i](x)
            outputs.append(x)

        return tf.keras.layers.concatenate(outputs, axis=3)


'''
class InceptionBlock(tf.keras.Model):
    def __init__(self, nb_filter_para):
        super(InceptionBlock, self).__init__()
        (branch1, branch2, branch3, branch4) = nb_filter_para
        self.branch1x1 = tf.keras.layers.Conv2D(branch1[0], (1, 1), padding='same', strides=(1, 1), name=None)

        self.branch3x3_1 = tf.keras.layers.Conv2D(branch2[0], (1, 1), padding='same', strides=(1, 1), name=None)
        self.branch3x3_2 = tf.keras.layers.Conv2D(branch2[1], (3, 3), padding='same', strides=(1, 1), name=None)

        self.branch5x5_1 = tf.keras.layers.Conv2D(branch3[0], (1, 1), padding='same', strides=(1, 1), name=None)
        self.branch5x5_2 = tf.keras.layers.Conv2D(branch3[1], (5, 5), padding='same', strides=(1, 1), name=None)

        self.branchpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branchpool_2 = tf.keras.layers.Conv2D(branch4[0], (1, 1), padding='same', strides=(1, 1), name=None)

    def call(self, inputs):
        b1 = self.branch1x1(inputs)
        b2 = self.branch3x3_1(inputs)
        b2 = self.branch3x3_2(b2)
        b3 = self.branch5x5_1(inputs)
        b3 = self.branch5x5_2(b3)
        b4 = self.branchpool_1(inputs)
        b4 = self.branchpool_2(b4)
        return tf.keras.layers.concatenate([b1, b2, b3, b4], axis=3)


def CustomInceptionModel_Test():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(InceptionBlock([(64,), (96, 128), (16, 32), (32,)]))
    model.add(InceptionBlock([(128,), (128, 192), (32, 96), (64,)]))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(2, 2), padding='same'))
    return model
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


class CustomModelForTest(tf.keras.Model):
    def __init__(self):
        super(CustomModelForTest, self).__init__()
        self.con1 = tf.keras.layers.Conv2D(8, 3, activation='relu')
        self.con2 = tf.keras.layers.Conv2D(16, 3, activation='relu')
        self.con3 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.res = ResBlock(32, strides=(2, 2))

    def call(self, inputs):
        x = self.con1(inputs)
        x = self.con2(x)
        x = self.con3(x)
        x = self.res(x)
        return x


class Classifier(tf.keras.Model):
    def __init__(self, classes):
        super(Classifier, self).__init__()
        self.flt = tf.keras.layers.Flatten()
        #self.drop = tf.keras.layers.Dropout(0.1)
        self.den1 = tf.keras.layers.Dense(classes)

    def call(self, inputs):
        x = self.flt(inputs)
        #x = self.drop(x)
        x = self.den1(x)
        return x
