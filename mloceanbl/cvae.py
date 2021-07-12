####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
import gp
tf.keras.backend.set_floatx('float64')


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

####################################################################################
########################### Define Models ##########################################
####################################################################################
 
class CVAE(keras.Model):
    r"""
    Implements a convolutional variational autoencoder for the relationship x = f(X) + b + noise, 
    where X = np.vstack(X[0], X[1], ...) with training model y = Lx + V, where L and V 
    are obtained via GP regression. f(X) is learned via a variational auto-encoder. 
    The auto encoder encodes and decodes the satellite sea surface data to train a 
    latent space on the gridded input data. From this latent space, we attach a third
    network that then predicts the mixed layer depth. 
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - latent_dim, variational latent space dimension
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
    def __init__(self, input_dim, n_features, x_l, latent_dim = 40, dtype='float64', **kwargs):
        super(CVAE, self).__init__(name='convolutional_variational_autoencoder', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.x_dim, self.y_dim = input_dim
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)

        
        if self.x_dim == 40:
            strides = [(2,3), (2,2), (2,2)]
        else:
            strides = [(2,4), (1, 3), (2,2)]
        
        self.prior1 = tfd.Independent(tfd.Normal(loc=tf.zeros(self.latent_dim, dtype =  'float64'), scale=1),
                        reinterpreted_batch_ndims=1)
        self.prior2 = tfd.Independent(tfd.Normal(loc=tf.zeros(self.x_dim*self.y_dim, dtype =  'float64'), scale=1),
                        reinterpreted_batch_ndims=1)
        
        self.encoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=(self.x_dim, self.y_dim, self.n_features)),
            tfkl.Conv2D(8, strides[0], strides = 1, padding = 'same'),
            tfkl.Conv2D(8, 3, strides = strides[0], padding = 'same'),
            tfkl.LayerNormalization(),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2D(16, strides[1], strides = 1, padding = 'same'),
            tfkl.Conv2D(16, 3, strides = strides[1], padding = 'same'),
            tfkl.LayerNormalization(),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2D(64, strides[2], strides = 1, padding = 'same'),
            tfkl.Conv2D(64, 3, strides = strides[2], padding = 'same'),
            tfkl.LayerNormalization(),
            tf.keras.layers.LeakyReLU(),
            tfkl.Flatten(),
            tfkl.Dense(2*self.latent_dim,
                   activation=None),
        ])
        
        self.decoder = tfk.Sequential([
            tfkl.InputLayer(input_shape=[self.latent_dim]),
            tfkl.Dense(5*5*64,
                   activation=None),
            tfkl.Reshape([5, 5, 64]),
            tfkl.Conv2DTranspose(64, strides[2],
                         padding='same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(64, 3, strides = strides[2], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(16, strides[1], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
          tfkl.Conv2DTranspose(16, 3, strides = strides[1], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(8, strides[0], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(8, 3, strides = strides[0], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(3, 1, padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Reshape([self.x_dim, self.y_dim, self.n_features])
            ])
        
        self.regressor = tfk.Sequential([
            tfkl.InputLayer(input_shape=[self.latent_dim]),
            tfkl.Dense(5*5*64,
                   activation=None),
            tfkl.Reshape([5, 5, 64]),
            tfkl.Conv2DTranspose(64, strides[2],
                         padding='same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(64, 3, strides = strides[2], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(16, strides[1], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
          tfkl.Conv2DTranspose(16, 3, strides = strides[1], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(8, strides[0], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(8, 3, strides = strides[0], padding = 'same'),
            tf.keras.layers.LeakyReLU(),
            tfkl.Conv2DTranspose(2, 1, padding = 'same'),
            ])

        
    def call(self, X_d, training = False):
        
        ## ANN Map
        self.X_d = tf.cast(tf.reshape(X_d, [-1,self.x_dim, self.y_dim, self.n_features]), dtype='float64')
        self.mean, self.logvar = tf.split(self.encoder(self.X_d, training = training), num_or_size_splits=2, axis=1)
        self.z = self.mean + tf.random.normal(tf.shape(self.mean), mean = self.mean, stddev = tf.exp(-0.5*self.logvar), dtype = 'float64')
        self.xp = self.decoder(self.z, training = training)
        
        x_v = self.regressor(self.z, training = training)
        x = tf.reshape(x_v[...,  0], [-1, self.x_dim, self.y_dim])
        x = tf.squeeze(x)
        self.var = tf.reshape(x_v[..., 1], [-1, self.x_dim, self.y_dim])
        self.var = 1e-8 + tf.math.softplus(self.var)
        self.var = tf.squeeze(self.var)
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
        loss = 1e-2*(d_mean + d_var - tf.math.log(d_var))
        loss += tf.reduce_mean( tf.abs(y_true - self.m))
        loss += tf.reduce_mean(tf.exp(self.logvar)+self.mean**2-self.logvar)
        loss += tf.reduce_mean( (self.X_d - self.xp)**2)
        
        return loss