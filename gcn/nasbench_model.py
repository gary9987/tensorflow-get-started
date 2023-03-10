import tensorflow as tf
import tensorflow.keras.backend as Keras_backend
import tensorflow.keras.layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import ECCConv, GINConvBatch, GATConv, GlobalSumPool, GlobalMaxPool, GlobalAvgPool, DiffPool


class GNN_Model(Model):

    def __init__(self, n_hidden, mlp_hidden, activation: str, dropout=0.):
        super(GNN_Model, self).__init__()
        #self.graph_conv = ECCConv(n_hidden, activation=activation)
        self.graph_conv = GINConvBatch(n_hidden, mlp_hidden=mlp_hidden, mlp_activation=activation, mlp_batchnorm=True,
                                       activation=activation)
        # self.graph_conv = GATConv(n_hidden, attn_heads=1, activation=activation)
        self.bn = tensorflow.keras.layers.BatchNormalization()
        self.pool = GlobalMaxPool()
        self.dropout = tensorflow.keras.layers.Dropout(dropout)
        self.dense = Dense(1)  # valid_acc

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.bn(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = self.dense(out)
        return out

'''
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 ecc_conv (ECCConv)          multiple                  12544     

 batch_normalization (BatchN  multiple                 1024      
 ormalization)                                                   

 global_max_pool (GlobalMaxP  multiple                 0         
 ool)                                                            

 dropout (Dropout)           multiple                  0         

 dense (Dense)               multiple                  771       

=================================================================
Total params: 14,339
Trainable params: 13,827
Non-trainable params: 512
_________________________________________________________________
'''

def get_weighted_mse_loss_func(mid_point, alpha):
    def weighted_mse(y_true, y_pred):
        scale_mse_loss = Keras_backend.switch((y_true <= mid_point), alpha * tf.square(y_true - y_pred), tf.square(y_true - y_pred))
        return tf.reduce_mean(scale_mse_loss, axis=-1)

    return weighted_mse


def bpr_loss(y_true, y_pred):

    N = tf.shape(y_true)[0]  # y_true.shape[0] = batch size
    lc_length = tf.shape(y_true)[1]

    total_loss = tf.constant([])

    for i in range(lc_length):
        tf.autograph.experimental.set_loop_options(
            shape_invariants=[(total_loss, tf.TensorShape([None]))]
        )
        loss_value = 0.0
        for j in range(N):
            loss_value += tf.reduce_sum(tf.keras.backend.switch(y_true[:, i] > y_true[j, i],
                                                                -tf.math.log(tf.sigmoid(y_pred[:, i] - y_pred[j, i])),
                                                                0))
        total_loss = tf.concat([total_loss, tf.expand_dims(loss_value, 0)], 0)

    return total_loss / tf.cast(N, tf.float32) ** 2


def is_weight_dir(filename):
    check_list = ['ecc_conv', 'gin_conv', 'gat_conv']
    for i in check_list:
        if i in filename:
            return True

    return False