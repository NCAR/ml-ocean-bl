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

class VCNN(keras.Model):
    r"""
    Implements convolutional neural network for the model d = f(S, T, H; theta)[0] + \epsilon
    and estimates the variance using as an additional output of the layer var = f(S, T, H; theta)[1],
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
        super(VCNN, self).__init__(name='variational_convolutional_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.x_dim, self.y_dim = input_dim
        self.n_features = n_features
        self.num_layers = 5
                
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)
        
        
       # Change kernel size depending on location  (EPO vs. SIO)
        if self.x_dim == 40:
            strides = (2,3)
        else:
            strides = (2,4)
        
        self.cnn = [tf.keras.layers.Conv2D(filters = min(128, 16*2**j), kernel_size = strides, strides = 1, padding = 'same') for j in range(self.num_layers)]
        self.cnn_final = tf.keras.layers.Conv2D(filters = 2, kernel_size = strides, strides = 1, padding = 'same')
        
        self.batch_norm = [tf.keras.layers.BatchNormalization() for j in range(self.num_layers)]
        self.leaky_relu = tf.keras.layers.LeakyReLU()
    def call(self, x, training=False):
        shape = tf.shape(x)
        if tf.size(shape) == 3:
            x = tf.expand_dims(x, 0)
        ## ANN Map
        for j in range(self.num_layers):
            x = self.cnn[j](x, training = training)
            x = self.leaky_relu(x, training = training)
            x = self.batch_norm[j](x, training = training)
        x = self.cnn_final(x)
        
        mean = tf.reshape(x[..., 0], (-1, self.x_dim, self.y_dim))
        self.var = 1e-8 + tf.math.softplus( tf.reshape( x[..., 1], (-1, self.x_dim, self.y_dim)) )
        self.var = tf.squeeze(self.var)
        return tf.squeeze(mean)
    
    
    def log_prob(self, y_pred, y_l, y_true, var = None, batch_num = None, training = True):
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