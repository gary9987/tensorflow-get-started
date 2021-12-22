import tensorflow as tf

#Define convolution with batchnromalization
def Conv2d_BN(x, nb_filter,kernel_size, padding='same',strides=(1,1),name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = tf.keras.layers.Conv2D(nb_filter,kernel_size,padding=padding,strides=strides,activation='relu',name=conv_name)(x)
    x = tf.keras.layers.BatchNormalization(axis=3,name=bn_name)(x)
    return x

#Define Inception structure
def Inception(x,nb_filter_para):
    (branch1,branch2,branch3,branch4)= nb_filter_para
    branch1x1 = tf.keras.layers.Conv2D(branch1[0],(1,1), padding='same',strides=(1,1),name=None)(x)

    branch3x3 = tf.keras.layers.Conv2D(branch2[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch3x3 = tf.keras.layers.Conv2D(branch2[1],(3,3), padding='same',strides=(1,1),name=None)(branch3x3)

    branch5x5 = tf.keras.layers.Conv2D(branch3[0],(1,1), padding='same',strides=(1,1),name=None)(x)
    branch5x5 = tf.keras.layers.Conv2D(branch3[1],(1,1), padding='same',strides=(1,1),name=None)(branch5x5)

    branchpool = tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
    branchpool = tf.keras.layers.Conv2D(branch4[0],(1,1),padding='same',strides=(1,1),name=None)(branchpool)

    x = tf.keras.layers.concatenate([branch1x1,branch3x3,branch5x5,branchpool],axis=3)

    return x


# Build InceptionV1 model
def InceptionV1(width, height, depth, classes):
    inpt = tf.keras.layers.Input(shape=(width, height, depth))

    x = Conv2d_BN(inpt, 64, (7, 7), strides=(2, 2), padding='same')
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2d_BN(x, 192, (3, 3), strides=(1, 1), padding='same')
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    x = Inception(x, [(64,), (96, 128), (16, 32), (32,)])  # Inception 3a 28x28x256
    x = Inception(x, [(128,), (128, 192), (32, 96), (64,)])  # Inception 3b 28x28x480
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 14x14x480

    x = Inception(x, [(192,), (96, 208), (16, 48), (64,)])  # Inception 4a 14x14x512
    x = Inception(x, [(160,), (112, 224), (24, 64), (64,)])  # Inception 4a 14x14x512
    x = Inception(x, [(128,), (128, 256), (24, 64), (64,)])  # Inception 4a 14x14x512
    x = Inception(x, [(112,), (144, 288), (32, 64), (64,)])  # Inception 4a 14x14x528
    x = Inception(x, [(256,), (160, 320), (32, 128), (128,)])  # Inception 4a 14x14x832
    x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 7x7x832

    x = Inception(x, [(256,), (160, 320), (32, 128), (128,)])  # Inception 5a 7x7x832
    x = Inception(x, [(384,), (192, 384), (48, 128), (128,)])  # Inception 5b 7x7x1024

    # Using AveragePooling replace flatten
    x = tf.keras.layers.AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=inpt, outputs=x)

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
    def __init__(self):
        super(Classifier, self).__init__()
        self.flt = tf.keras.layers.Flatten()
        self.drop = tf.keras.layers.Dropout(0.1)
        self.den1 = tf.keras.layers.Dense(10)


    def call(self, inputs):
        x = self.flt(inputs)
        x = self.drop(x)
        x = self.den1(x)
        return x
