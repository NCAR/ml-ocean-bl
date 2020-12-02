####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions
tfb = tfp.bijectors
import gp

####################################################################################
########################### Define Models ##########################################
####################################################################################
       
class VANN(keras.Model):
    r"""
    Implements an artificial neural network for the relationship x = f(X) + noise, 
    where X = np.vstack(X[0], X[1], ...) with training model y = Lx + V, where L and V 
    are obtained via GP regression. Noise is estimated by parameterization via a 
    Gaussian distribution at the output layer.
    
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
    def __init__(self, input_dim, n_features, x_l, l1_ = 1e-4, dtype='float64', **kwargs):
        super(VANN, self).__init__(name='variational_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)
        
        l1 = tf.keras.regularizers.l1(0.01*l1_)
        l2 = tf.keras.regularizers.l2(0.001*l1_)
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
            tf.keras.layers.Dense(
                self.input_dim, 
                input_shape=(self.input_dim,),
                activation='relu',
                kernel_regularizer = l2,
                bias_regularizer = l2
                )
        )
        self.L.add(
            tf.keras.layers.Dense(
                2*self.input_dim, 
                input_shape=(self.input_dim,),
                # activation='relu',
                kernel_regularizer = l2,
                bias_regularizer = l2,
                )
        )
        self.L.add(
            tfp.layers.DistributionLambda(
                lambda t: tfd.MultivariateNormalDiag(loc=t[..., :self.input_dim],
                           scale_diag= np.float64(1e-8) + tf.math.softplus(t[...,self.input_dim:]))),
        )
    
        
    def call(self, X_d, training=True):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        self.x_dist = self.L(tf.cast(tf.reshape(X_d, [1,-1]), dtype='float64'), training=training)
        x = self.x_dist.mean()

        x = tf.reshape(x, (-1,))
        self.noise = tf.reshape( self.x_dist.variance(), (-1,))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l, noise = self.noise)
        self.sample = self.gp.sample()
        loss = -self.gp.log_prob( y_true )
        loss += tf.math.reduce_mean(self.L.losses)
        return loss
