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
        self.dense = Dense(3)  # train_acc, valid_acc, test_acc

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
        scale_mse_loss = Keras_backend.switch((y_true < mid_point), alpha * tf.square(y_true - y_pred), tf.square(y_true - y_pred))
        return tf.reduce_mean(scale_mse_loss, axis=-1)

    return weighted_mse