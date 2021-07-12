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

class Linear(keras.Model):
    r"""
    Implements linear model x = \sum_{n=1}^{num_features} w[n].X[n] + b + noise
    with training model y = Lx + V, where L and V are obtained via GP regression.
    Input noise is estimated along with parameters w and b.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - x_l,  locations of inputs, shape (input_dim, 2)
                            columns house (lon,lat) coordinates respectively.
                    - y_l, locations of training data
                    - y_d, training data values
                            
    Output arguments - x, estimate of x = f(X) + noise
                     - m, mean estimate of m = Lx + V
                     - v, diagonal variance of gp covariance V
                     
    Parameters       - w, linear weight matrix, shape (input_dim, n_features)
                     - b, bias, shape (input_dim, )
                     
    
    Inherited Parameters - input_noise, input-dependent noise estimate, shape (input_dim,)
                         gives estimate of variances
                         - .gp.amplitude, kernel amplitude
                         - .gp.length_Scale, kernel length scale, 1 / correlation distance
    
    
    """
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(Linear, self).__init__(name='linear_projection', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.x_dim, self.y_dim = input_dim
        self.n_features = n_features
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)
        
        
        # Parameters, w, b
        # Linear weights
        w_init = tf.initializers.GlorotNormal()
        self.A1 = tf.Variable(initial_value=w_init(shape=(self.x_dim, self.y_dim, self.n_features),
                                    dtype='float64'),trainable=True, 
                            name = 'linear')
        # bias weights
        b_init = tf.initializers.GlorotNormal()
        self.b1 = tf.Variable(initial_value=b_init(shape=(self.x_dim, self.y_dim),
                                    dtype='float64'), trainable=True, 
                            name = 'bias')
        
        self.A2 = tf.Variable(
                initial_value=w_init(shape = (self.x_dim, self.y_dim, self.n_features)), trainable=True, validate_shape=True, caching_device=None,
                name='Weight_matrix_2', dtype='float64')
        
        self.b2 = tf.Variable(
                initial_value=b_init(shape = (self.x_dim, self.y_dim)), trainable=True, validate_shape=True, caching_device=None,
                name='bias_matrix_2', dtype='float64')

        
    def call(self, inputs):

        ## Linear Map
        x = tf.math.multiply(inputs, self.A1)
        x = tf.math.reduce_sum(x, axis=-1) + self.b1
        
        self.var = tf.math.multiply(self.A2, inputs)
        self.var = tf.reduce_sum(self.var, axis = -1) + self.b2
        self.var = 1e-8 + tf.math.softplus(self.var)
        
        return x
    
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
        d = (y_true - self.m)**2 / self.v
        d_mean = tf.reduce_mean(d)
        d_var = tf.math.reduce_variance(d)
        loss = 1e-1*(d_mean + d_var - tf.math.log(d_var))
        loss += tf.reduce_mean( tf.abs(y_true - self.m))
        loss += 1e-1*tf.reduce_mean(self.v*(self.v-1))
        return loss