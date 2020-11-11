####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions
tfb = tfp.bijectors

####################################################################################
########################### Define Models ##########################################
####################################################################################
 
class VAE(keras.Model):
    r"""
    Implements a variational autoencoder for the relationship x = f(X) + b + noise, 
    where X = np.vstack(X[0], X[1], ...) with training model y = Lx + V, where L and V 
    are obtained via GP regression.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - x_l,  locations of inputs, shape (input_dim, 2)
                            columns house (lon,lat) coordinates respectively.
                    - y_l, locations of training data
                    - y_d, training data values
                            
    Output arguments - x, estimate of x = f(X) + noise
                     - m, mean estimate of m = Lx + V
                     - v, diagonal variance of gp covariance V
                     
    Parameters       - w_i, linear weight matrix, shape (input_dim, n_features)
                     - b_i, bias, shape (input_dim, )
    
    Inherited Parameters - input_noise, input-dependent noise estimate, shape (input_dim,)
                         gives estimate of variances 
                         - .gp.amplitude, kernel amplitude
                         - .gp.length_Scale, kernel length scale
    
    
    """
    def __init__(self, input_dim, n_features, x_l, latent_dim = 40, l1_regularizer = 1e-6, dtype='float64', **kwargs):
        super(VAE, self).__init__(name='variational_autoencoder', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)
        
        l1 = tf.keras.regularizers.l1(
                l1_regularizer,
                )
        l2 = tf.keras.regularizers.l2(
                l1_regularizer,
                )
        
        # Parameters, w, b, input_noise
        # Linear weights
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.n_features*self.input_dim,),
            tf.keras.layers.Dense(self.input_dim, 
                                  activation='relu',
                                 kernel_regularizer = l1),
            tf.keras.layers.Dense(self.latent_dim + self.latent_dim,),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.latent_dim,),
            tf.keras.layers.Dense(self.input_dim, activation='relu',),
            tf.keras.layers.Dense(self.input_dim*self.n_features),
        ])

        self.regressor = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.latent_dim,),
            tf.keras.layers.Dense(self.input_dim, activation='relu',
                                 activity_regularizer = l2),
            tf.keras.layers.Dense(self.input_dim, activation='relu',
                                 activity_regularizer = l2),
            tf.keras.layers.Dense(self.input_dim),
        ])

 

    @tf.function
    def funcsample(self, mean, logvar, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim), dtype='float64')
        return self.regressor(eps * tf.exp(logvar*0.5) + mean, training=True)
    
    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape = mean.shape, dtype='float64')
        return eps * tf.exp(logvar*0.5) + mean
    
    def decode(self, z):
        x = self.decoder(z)
        return x

        
    def call(self, X_d):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        self.X_d = tf.cast(tf.reshape(X_d, [1,-1]), dtype='float64')
        self.mean, self.logvar = self.encode(self.X_d)
        self.z = self.reparameterize(self.mean, self.logvar)
        self.xp = self.decode(self.z)
        x_samp = self.funcsample(self.mean, self.logvar)
        x = tf.reduce_mean(x_samp, axis=0)
        self.noise = tf.reshape(tf.reduce_mean( (x_samp - x)**2, axis=0 ), (-1,))
        x = tf.reshape(x, (-1, ))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l, noise = self.noise)
        self.sample = self.gp.sample()
        loss = -self.gp.log_prob( y_true )
        loss += tf.math.reduce_sum((self.X_d - self.xp)**2)
        loss += tf.math.reduce_sum( self.z**2 )
        loss -= tf.math.reduce_sum( (self.z - self.mean)**2*tf.exp(-self.logvar) + self.logvar )
        return loss