####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions
tfb = tfp.bijectors
import gp

####################################################################################
########################### Define Models ##########################################
####################################################################################
    
class ANN_dropout(keras.Model):
    r"""
    Implements an artificial neural network for the relationship x = f(X) + noise, 
    where X = np.vstack(X[0], X[1], ...) with training model y = Lx + V, where L and V 
    are obtained via GP regression. Dropout is implemented to help estimates noise.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - x_l,  locations of inputs, shape (input_dim, 2)
                            columns house (lon,lat) coordinates respectively.
                    - y_l, locations of training data
                    - y_d, training data values
                            
    Output arguments - x, estimate of x = f(X) + noise
                     - m, mean estimate of m = Lx + V
                     - v, diagonal variance of gp covariance V
                     
    
    Inherited Parameters - input_noise, input-dependent noise estimate, shape (input_dim,)
                         gives estimate of variances 
                         - .gp.amplitude, kernel amplitude
                         - .gp.length_Scale, kernel length scale
    
    
    """
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(ANN_dropout, self).__init__(name='dropout_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)
        
        l1 = tf.keras.regularizers.l2(1e-8)
        
        # Parameters, w, b, input_noise
        # Linear weights
        self.L = tf.keras.Sequential()
        self.L.add(
            tf.keras.layers.Dense(
                self.input_dim, 
                input_shape=(self.n_features*self.input_dim,),
                activation='relu',
                kernel_regularizer = l1,
                bias_regularizer = l1
                )
        )
        self.L.add(
            tf.keras.layers.Dropout(0.2)
        )
        self.L.add(
            tf.keras.layers.Dense(
                self.input_dim, 
                input_shape=(self.input_dim,),
                activation='relu',
                kernel_regularizer = l1,
                bias_regularizer = l1
                )
        )
        self.L.add(
            tf.keras.layers.Dropout(0.5)
        )
        self.L.add(
            tf.keras.layers.Dense(
                self.input_dim, 
                input_shape=(self.input_dim,),
                )
        )
    
    def monte_carlo_sample(self, X_d, training = True):
        eps = 1e-2*tf.random.normal(shape=(50, self.input_dim*self.n_features), dtype='float64')
        z = X_d + eps
        return self.L(z, training=training)
        
    def call(self, X_d, training=True):       
        ## ANN Map
        x_samp = self.monte_carlo_sample(tf.cast(tf.reshape(X_d, [1,-1]), dtype='float64'), training=True)
        x = tf.math.reduce_mean(x_samp, axis=0)
        self.var = tf.math.reduce_mean((x - x_samp), axis=0)
        return x
    
    def log_prob(self, y_pred, y_l, y_true, var = None, batch_num = None, training = True):
        r"""
        Computes the gaussian process log-likelihood of the data given
        the estimate y_pred (or x from self.call(...) ). self.sample
        is produced for other metric purposes (see ./train.py)
        """
        if var is None:
            var = self.var
        self.m, self.v = self.gp(y_pred, y_l, noise = var, training = training)
        self.sample = self.gp.sample()
        loss = -self.gp.log_prob( y_true )
        loss += tf.math.reduce_mean(self.L.losses)
        return loss
    