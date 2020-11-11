####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
from netCDF4 import Dataset
import tensorflow as tf
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
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)
        
        
        # Parameters, w, b
        # Linear weights
        w_init = tf.initializers.GlorotNormal()
        self.w = tf.Variable(initial_value=w_init(shape=(self.input_dim, self.n_features),
                                    dtype='float64'),trainable=True, 
                            name = 'linear')
        
        # bias weights
        b_init = tf.initializers.GlorotNormal()
        self.b = tf.Variable(initial_value=b_init(shape=(self.input_dim,),
                                    dtype='float64'), trainable=True, 
                            name = 'bias')
        
    def call(self, x):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        ## Linear Map
        x = tf.math.multiply(x, self.w)
        x = tf.math.reduce_sum(x, axis=1) + self.b
        x = tf.reshape(x, (-1,))
        
        return x
    
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l)
        self.sample = self.gp.sample()
        loss = -self.gp.log_prob( y_true )
        return loss