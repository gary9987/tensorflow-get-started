import tensorflow as tf

class CustomBranch(tf.keras.Model):
    def __init__(self, branch_par = [['Conv2D 64 1'], ['Conv2D 96 1', 'Conv2d 128 3']]):
        super(CustomBranch, self).__init__()

        self.branch_list = []

        for branch in branch_par:

            a_branch = []

            for layer in branch:
                layer_type, filters, kernal_size = layer.split()
                filters = int(filters)
                kernal_size = (int(kernal_size), int(kernal_size))
                if layer_type == 'Conv2D':
                    a_branch.append(tf.keras.layers.Conv2D(filters, kernal_size, padding='same', strides=(1, 1)))

            self.branch_list.append(a_branch)


        '''
        self.branch1x1 = tf.keras.layers.Conv2D(branch1[0], (1, 1), padding='same', strides=(1, 1), name=None)
        self.branchpool_1 = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')
        self.branchpool_2 = tf.keras.layers.Conv2D(branch4[0], (1, 1), padding='same', strides=(1, 1), name=None)
        '''

    def call(self, inputs):
        '''
        b1 = self.branchpool_1(inputs)
        b2 = self.branch3x3_1(inputs)
        b2 = self.branch3x3_2(b2)
        b3 = self.branch5x5_1(inputs)
        b3 = self.branch5x5_2(b3)
        b4 = self.branchpool_1(inputs)
        b4 = self.branchpool_2(b4)

        return tf.keras.layers.concatenate([b1, b2, b3, b4], axis=3)
        '''
        return True


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
        b1 = self.branchpool_1(inputs)
        b2 = self.branch3x3_1(inputs)
        b2 = self.branch3x3_2(b2)
        b3 = self.branch5x5_1(inputs)
        b3 = self.branch5x5_2(b3)
        b4 = self.branchpool_1(inputs)
        b4 = self.branchpool_2(b4)
        return tf.keras.layers.concatenate([b1, b2, b3, b4], axis=3)


def CustomInceptionModel_test():
    model = tf.keras.Sequential()
    model.add(CustomBranch())
    return model


def CustomInceptionModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization(axis=3))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(InceptionBlock([(64,), (96, 128), (16, 32), (32,)]))
    model.add(InceptionBlock([(128,), (128, 192), (32, 96), (64,)]))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(2, 2), padding='same'))
    """
    [[b1], [b2], [b3]]
    """
    return model


class CustomModel(tf.keras.Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.con1 = tf.keras.layers.Conv2D(8, 3, activation='relu')
        self.con2 = tf.keras.layers.Conv2D(16, 3, activation='relu')
        self.con3 = tf.keras.layers.Conv2D(32, 3, activation='relu')

    def call(self, inputs):
        x = self.con1(inputs)
        x = self.con2(x)
        x = self.con3(x)
        return x


class Classifier(tf.keras.Model):
    def __init__(self, classes):
        super(Classifier, self).__init__()
        self.flt = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(0.1)
        self.den1 = tf.keras.layers.Dense(classes)

    def call(self, inputs):
        x = self.flt(inputs)
        x = self.drop(x)
        x = self.den1(x)
        return x
