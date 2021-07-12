####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

tf.keras.backend.set_floatx('float64')
import gp


####################################################################################
########################### Define Models ##########################################
####################################################################################

class VLCNN(keras.Model):
    r"""
    Implements convolutional neural network for the model d = f(S, T, H; theta) + \epsilon
    and estimates the variance using a linear model x = \sum_{n=1}^{num_features} w[n].X[n] + b,
    with training model y = Lx + V, where L and V are obtained via GP regression.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - x_l,  locations of inputs, shape (input_dim, 2)
                            columns house (lon,lat) coordinates respectively.
                    - y_l, locations of training data
                    - y_d, training data values
                            
    Output arguments - x, estimate of x = f(X) + noise 
                     - var, variance estimate of noise
                     - m, mean estimate of m = Lx + V
                     - v, diagonal variance of gp covariance V
                        
    
    """
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(VLCNN, self).__init__(name='variational_linear_convolutional_neural_network', dtype='float64', **kwargs)
        
        # Sizes and setup
        num_dense_layers = 0
        self.num_dense_layers = 0
        self.input_dim = input_dim
        self.x_dim, self.y_dim = input_dim
        self.n_features = n_features
        num_layers = 5
        self.num_layers = num_layers
        batch_bool = True
        self.batch_bool = batch_bool
        Dropout = 0.1
        self.Dropout = Dropout
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)
        
        # Alter size of kernels
        if self.x_dim == 40:
            location = "EPO"
        else:
            location = "SIO"
        
        if num_dense_layers == 0:
            if location == "EPO":
                strides = (2,3)
            else:
                strides = (2,4)
            self.cnn = [tf.keras.layers.Conv2D(filters = min(16*2**j, 128), kernel_size = strides, strides = 1, padding = 'same') for j in range(num_layers)]
            self.cnn_final = tf.keras.layers.Conv2D(filters = 1, kernel_size = strides, strides = 1, padding = 'same')
        else:
            if location == "EPO":
                strides = [(2,3), (2,2), (2,2)]
            else:
                strides = [(2,4), (2,3), (1,2)]
            self.cnn = [tf.keras.layers.Conv2D(filters = min(16*2**j, 128), kernel_size = 3, strides = strides[j], padding = 'same') for j in range(num_layers)]
            self.dense_layers = [tf.keras.layers.Dense( int(self.x_dim*self.y_dim / 2**(num_dense_layers-j-1) ) ) for j in range(num_dense_layers)]
            self.flatten = tf.keras.layers.Flatten()
        self.batch_norm = [tf.keras.layers.BatchNormalization() for j in range(num_layers)]
        self.dropout = [tf.keras.layers.Dropout(Dropout) for j in range(num_layers)]
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        
        
        # Linear variance estimator
        initializer = tf.keras.initializers.GlorotNormal()
        self.A2 = tf.Variable(
                initial_value=initializer(shape = (self.x_dim, self.y_dim, self.n_features)), trainable=True, validate_shape=True, caching_device=None,
                name='Weight_matrix_2', dtype='float64')
        
        self.b2 = tf.Variable(
                initial_value=initializer(shape = (self.x_dim, self.y_dim)), trainable=True, validate_shape=True, caching_device=None,
                name='bias_matrix_2', dtype='float64')

    def call(self, inputs, training=False):
        x = tf.identity(inputs)
        shape = tf.shape(x)
        if tf.size(shape) == 3:
            x = tf.expand_dims(x, 0)
        for j in range(self.num_layers):
            x = self.cnn[j](x, training = training)
            x = self.leaky_relu(x, training = training)
            if self.batch_bool:
                x = self.batch_norm[j](x, training = training)
            x = self.dropout[j](x, training = training)
        if self.num_dense_layers == 0:
            x = self.cnn_final(x)      
            mean = tf.reshape(x[..., 0], (-1, self.x_dim, self.y_dim))
        else:
            x = self.flatten(x)
            for j in range(self.num_dense_layers):
                x = self.dense_layers[j](x)
                x = self.leaky_relu(x)
            mean = tf.reshape(x, (-1, self.x_dim, self.y_dim))
        
        
        self.var = tf.math.multiply(self.A2, inputs)
        self.var = tf.reduce_sum(self.var, axis = -1) + self.b2
        self.var = 1e-8 + tf.math.softplus(self.var)
        
        self.var = tf.squeeze(self.var)
        return tf.squeeze(mean)
    
    
    def log_prob(self, y_pred, y_l, y_true, var = None, batch_num = None, training = True):
        r"""
        Computes the gaussian process log-likelihood of the data given
        the estimate y_pred (or x from self.call(...) ). self.sample
        is produced for other metric purposes (see ./train.py)
        """
        if var is None:
            var = self.var
        y_pred = tf.reshape(y_pred, (-1, 1))
        self.m, self.v = self.gp(y_pred, y_l, var, training = training)
        self.m = tf.reshape(self.m, (-1,))
        self.sample = self.gp.sample()
#         loss = -self.gp.log_prob( y_true )
        d = (y_true - self.m)**2 / self.v
        d_mean = tf.reduce_mean(d)
        d_var = tf.math.reduce_variance(d)
        loss = 1e-1*(d_mean + d_var - tf.math.log(d_var))
        loss += tf.reduce_mean( tf.abs(y_true - self.m))
        loss += 1e-1*tf.reduce_mean(self.v*(self.v-1))
        return loss