####################################################################################
####################################################################################
########################### File Description #######################################
####################################################################################
####################################################################################

# This file creates a class that imports preprocessed sss data, sst data, and ssh data 
# It currently hosts a number of plot creation functions. It will hold all of the code
# for the potential models that we will test.

## TO DO
# Add argo data
# Add models
# Add plot creation


####################################################################################
########################### Import libraries #######################################
####################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import seaborn as sns
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.basemap import maskoceans
import tensorflow as tf
import tensorflow.keras as keras
import gpflow as gpf
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions
tfb = tfp.bijectors

####################################################################################
########################### Import Data ############################################
####################################################################################

            
class dataset:
    def __init__(self, x_data, lon, lat, y_data, m_lon, m_lat, lat_bounds, lon_bounds):
        self.X_data = np.stack(x_data, axis=-1)
        lat_mask = (lat < lat_bounds[1]) & (lat > lat_bounds[0])
        lon_mask = (lon < lon_bounds[1]) & (lon > lon_bounds[0])
        self.mask = np.isfinite(x_data[0].mean(axis=0))
        self.mask[~lat_mask, :] = False
        self.mask[:, ~lon_mask] = False
        self.X_data = np.stack(x_data, axis=-1)[:, self.mask, :]
        
        [LAT, LON] = np.meshgrid(lat, lon, indexing='ij')
        self.X_loc = np.stack((LON[self.mask], LAT[self.mask]), axis=-1)
        self.y_data = y_data
        self.y_loc = np.stack((m_lon, m_lat), axis=-1)
        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        
    def normalize(self, train_index):
        
        for i in range(self.X_data.shape[-1]):
            self.X_data[:, :, i] = normalize(self.X_data[:, :, i], train_index)
        
        self.y_data = normalize(self.y_data, train_index)
        
    def get_index(self, index):
        mask = (self.y_loc[index, :, 1]<self.lat_bounds[1]) & (self.y_loc[index, :, 1]>self.lat_bounds[0]) & \
                (self.y_loc[index, :, 0]<self.lon_bounds[1]) & (self.y_loc[index, :, 0]>self.lon_bounds[0])

        y = self.y_data[index, mask].reshape(-1, 1)
        y_loc = self.y_loc[index, mask, :]
        return self.X_data[index], self.X_loc, y, y_loc
    
    def train_test_val_indices(self, n, n_train, n_test):
        n_validate = n - n_train -  n_test
        assert n_validate > 0, print('Validation size must be positive')
        i = np.arange(n)
        np.random.seed(0)
        np.random.shuffle(i)
        self.i_train = i[:n_train]
        self.i_test = i[n_train:n_train+n_test]
        self.i_validate = i[n_train+n_test:]
        return i[:n_train], i[n_train:n_train+n_test], i[n_train+n_test:]

    def calc_covariance(self, indicies):
        return np.var(self.X_data[indicies], axis = 0)

####################################################################################
########################### Preprocess Models ######################################
####################################################################################

def train_test_val_indices(n, n_train, n_test):
    n_validate = n - n_train -  n_test
    assert n_validate > 0, print('Validation size must be positive')
    i = np.arange(n)
    rng = np.random.default_rng(10)
    np.random.shuffle(i)
    return i[:n_train], i[n_train:n_train+n_test], i[n_train+n_test:]

def normalize(inputs, train_index):
    return (inputs - np.nanmean(inputs[train_index], axis=0))/np.nanstd(inputs[train_index], axis=0)



####################################################################################
########################### Define Models ##########################################
####################################################################################
    
class GPR(keras.Model):
    def __init__(self, x_l, dtype='float64', **kwargs):
        super(GPR, self).__init__(name='gaussian_process', dtype='float64', **kwargs)
        self.x_l = x_l
        self.input_dim = self.x_l.shape[0]

        self.amplitude = tf.Variable(np.float64(2.0), dtype='float64',
                                          trainable=True, name='amplitude')
        self.length_scale = tf.Variable(np.float64(1.0), dtype='float64', 
                                            trainable=True, name='length_scale')

        self.input_noise = tf.Variable(1e-4*tf.ones([self.input_dim,], dtype='float64'),
                                            trainable=True, name='input_noise')
    
    def call(self, x, y_l, noise = None):
        # Calculuates a gaussian process
        # m = Lx + V
        # where L = Kyx (Kxx + \sigmaI)^{-1}
        # where V = Kyy - L Kxy + L\sigmaIL^{T}
        
        self.train(x)
        Kxx = self.kernel(self.x_l)
        Kxy = self.kernel(self.x_l, y_l)
        Kyy = self.kernel(y_l, y_l)
        k_diag = tf.linalg.diag_part(Kxx)
        s_diag = tf.fill([self.input_dim], np.float64(1e-3))
        ks = tf.linalg.set_diag(Kxx, 
                k_diag + s_diag + tf.nn.softplus(self.input_noise))

        L = tf.linalg.cholesky(ks)
        m = tf.linalg.matmul(Kxy, 
            tf.linalg.cholesky_solve(L, x), 
            transpose_a = True
        )
        v = Kyy - tf.linalg.matmul(Kxy, 
            tf.linalg.cholesky_solve(L, Kxy), 
            transpose_a = True
        ) 
        if noise != None:
            noise_v = tf.linalg.matmul(Kxy,
                tf.linalg.cholesky_solve(L, 
                tf.linalg.diag(  tf.math.sqrt(tf.nn.softplus(noise)) )),
                transpose_a = True)
            noise_v = tf.linalg.matmul(noise_v, noise_v, transpose_b = True)
        
            return m, tf.linalg.diag_part(v+noise_v)
        else:
            return m, tf.linalg.diag_part(v)


    def train(self, x):
        opt = keras.optimizers.Adam(1e-3)
        self.x = x
        for j in range(5):
            opt.minimize(self.log_prob, self.trainable_variables)

    def log_prob(self):
        # Calculates the log probability for the
        # gaussian prior.
        
        K = self.kernel(self.x_l)
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([self.input_dim], np.float64(1e-4))
        ks = tf.linalg.set_diag(K, k_diag + s_diag + tf.nn.softplus(self.input_noise) )
        L = tf.linalg.cholesky(ks)
        d = tfd.MultivariateNormalTriL(loc = tf.zeros([self.input_dim], dtype='float64'), 
                                        scale_tril = L)
        l = -d.log_prob(tf.reshape(self.x, [1, self.input_dim]))
        d_a = tfd.LogNormal(loc=np.float64(0.0), scale = np.float64(1.0))
        l -= d_a.log_prob(self.amplitude)
        l -= d_a.log_prob(self.length_scale)
        d_g = tfd.Gamma(concentration = np.float64(2.0), 
                            rate = np.float64(2.0))
        l -= tf.math.reduce_mean(d_g.log_prob(self.input_noise))
        return l
        
    def convert_(self, x):
        xx = np.stack( (np.sin(np.deg2rad(90.0-x[:,1]))*np.cos(np.deg2rad(x[:,0]+180.0)),  
                     np.sin(np.deg2rad(90.0-x[:,1]))*np.sin(np.deg2rad(x[:,0]+180.0)),
                     np.cos(np.deg2rad(90.0-x[:,1]))), axis=-1)
        xx = tf.cast(xx, dtype='float64')
        return xx

    def kernel(self, X, X2=None):
        if X2 is None:
            X2 = X
        X = self.convert_(X)
        X2 = self.convert_(X2)
        dist = tf.linalg.matmul(X, X2, transpose_b=True)
        dist = tf.clip_by_value(dist, -1, 1) # Needed for numerical stability
        dist = 60*tf.math.acos(dist)
        return self.matern(dist)

    def matern(self, dist):
        z = tf.math.sqrt(np.float64(5.0)) * dist / self.length_scale
        z = self.amplitude*(np.float64(1.0) + z + (z ** 2) / 3) * tf.math.exp(-z)
        return z

class DenseIdentityBlock(keras.Model):
    def __init__(self,  output_dim, name='DenseIdentityBlock'):
        super(DenseIdentityBlock, self).__init__(name=name, dtype='float64')
        self.num_layers = num_layers
        self.dense1 = tf.keras.layers.Dense(output_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dim, activation='relu')
        self.dense3 = tf.keras.layers.Dense(output_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.1)
    def call(self, input_tensor, training=False):
        x = self.dense1(input_tensor)
        x = self.dense2(x)
        x += input_tensor + self.dense3(input_tensor)*tf.random.normal(input_tensor.shape, stddev = 0.01, dtype='float64')
        return x
    
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
        self.gp = GPR(x_l)
        
        
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
        x = tf.reshape(x, (-1, 1))
        
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        # Calculate log probability as a loss
        self.m, self.v = self.gp(y_pred, y_l)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.math.reduce_mean(tf.math.log(self.v))

        return loss
    
    
    def grad(self, X_d, y_l, y_d):
        # Calculate gradients manually
        knn = self.gp.kernel(self.x_l)
        s = tf.linalg.diag(tf.fill([self.input_dim], np.float64(1e-3)))
        s += tf.linalg.diag(tf.math.log(1+tf.math.exp(self.gp.input_noise)))
        kmn = self.gp.kernel(y_l, self.x_l)
        K = tf.linalg.matmul(kmn,
                tf.linalg.cholesky_solve(
                    tf.linalg.cholesky(knn+s), 
                    tf.eye(self.input_dim, dtype='float64')
                    )
                    )
        
        dJdb_temp = y_d.reshape(-1, 1) 
        dJdb_temp -= tf.linalg.matmul(K, 
                                      tf.reshape(
                                          tf.math.reduce_sum(tf.math.multiply(X_d, self.w), axis=1),
                                          [self.input_dim, 1])) 
        dJdb_temp -= tf.linalg.matmul(K, 
                                      tf.reshape(self.b, [self.input_dim,1]))
        dJdb_temp = tf.linalg.matmul(K, dJdb_temp, transpose_a = True)
        dJdb = -dJdb_temp
        dJdL = -tf.math.multiply(X_d, dJdb_temp)
        return dJdL, tf.reshape(dJdb, [self.input_dim,])
    
class DenseLinear(keras.Model):
    r"""
    Implements linear model x = LX + b + noise, where X = np.vstack(X[0], X[1], ...)
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
                         - .gp.length_scale, kernel length scale
    
    
    """
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(DenseLinear, self).__init__(name='dense_linear_projection', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        # Parameters, w, b, input_noise
        # Linear weights
        self.L = tf.keras.layers.Dense(self.input_dim, 
                                       input_shape=(self.n_features*self.input_dim,),
                                       kernel_regularizer = tf.keras.regularizers.l1(1e-2),
                                       bias_regularizer = tf.keras.regularizers.l1(1e-2)
                                      )
        
    def call(self, x):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """

        ## Linear Map
        x = self.L(tf.cast(tf.reshape(x, [1,-1]), dtype='float64'))
        x = tf.reshape(x, (-1, 1))
        
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        
        self.m, self.v = self.gp(y_pred, y_l)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.math.reduce_mean(tf.math.log(self.v))
        loss += tf.math.reduce_sum(self.L.losses)
        return loss
    
    
    def grad(self, X_d, y_d):
        # Code for manual gradients
        knn = self.k(self.x_l)
        s = tf.linalg.diag(tf.fill([self.input_dim], self.m.likelihood.variance))
        s += tf.linalg.diag(self.input_noise)
        kmn = self.k(self.y_l, self.x_l)
        K = tf.linalg.matmul(kmn,
                tf.linalg.cholesky_solve(
                    tf.linalg.cholesky(knn+s), 
                    tf.eye(self.input_dim, dtype='float64')
                    )
                    )

        dJdb_temp = y_d.reshape(-1, 1) 
        dJdb_temp -= tf.linalg.matmul(K, 
                                      tf.linalg.matmul(self.L.kernel, X_d.reshape(-1,1), transpose_a = True)) 
        dJdb_temp -= tf.linalg.matmul(K, 
                                      tf.reshape(self.L.bias, [self.input_dim,1]))
        dJdb_temp = tf.linalg.matmul(K, dJdb_temp, transpose_a = True)
        dJdb = -dJdb_temp
        dJdL = -tf.transpose(tf.linalg.matmul(dJdb_temp, X_d.reshape(-1,1), transpose_b = True))
        return dJdL, tf.reshape(dJdb, [self.input_dim,])
       
class ANN(keras.Model):
    r"""
    Implements an artificial neural network for the relationship x = f(X) + b + noise, 
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
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(ANN, self).__init__(name='neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        # l1 = tf.keras.regularizers.l1(1e-4)
        
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
                kernel_regularizer = l1,
                bias_regularizer = l1
                )
        )
        self.L.add(
            tf.keras.layers.Dense(
                self.input_dim, 
                input_shape=(self.input_dim,),
                )
        )
        
        
    def call(self, X_d):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        x = self.L(tf.cast(tf.reshape(X_d, [1,-1]), dtype='float64'), training=True)
        x = tf.reshape(x, (-1, 1))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.math.reduce_mean(tf.math.log(self.v))
        loss += tf.math.reduce_mean(self.L.losses)
        return loss

class ANN_dropout(keras.Model):
    r"""
    Implements an artificial neural network for the relationship x = f(X) + b + noise, 
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
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(ANN_dropout, self).__init__(name='dropout_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        # l1 = tf.keras.regularizers.l1(1e-4)
        
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
            tf.keras.layers.Dropout(0.1)
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
            tf.keras.layers.Dropout(0.1)
        )
        self.L.add(
            tf.keras.layers.Dense(
                self.input_dim, 
                input_shape=(self.input_dim,),
                )
        )
    
    def sample(self, X_d, training = True):
        eps = 1e-2*tf.random.normal(shape=(50, self.input_dim, self.n_features), dtype='float64')
        z = X_d + eps
        return self.L(z, training=training)
        
    def call(self, X_d):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        x_samp = self.sample(tf.cast(tf.reshape(X_d, [1,-1]), dtype='float64'))
        x = tf.math.reduce_mean(x_samp, axis=0)
        x = tf.reshape(x, (-1, 1))
        self.noise = tf.math.reduce_mean( (x - x_samp)**2, axis=0 )
        self.noise = tf.reshape(self.noise, (-1,))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l, self.noise)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.math.reduce_mean(tf.math.log(self.v))
        loss += tf.math.reduce_mean(self.L.losses)
        return loss
    
class ANN_distribution(keras.Model):
    r"""
    Implements an artificial neural network for the relationship x = f(X) + b + noise, 
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
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(ANN_distribution, self).__init__(name='dropout_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        # l1 = tf.keras.regularizers.l1(1e-4)
        
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
                kernel_regularizer = l1,
                bias_regularizer = l1
                )
        )

        self.L.add(
            tf.keras.layers.Dense(
                2*self.input_dim, 
                input_shape=(self.input_dim,),
                )
        )
    
        
    def call(self, X_d):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        x, x_noise = tf.split(self.L(tf.cast(tf.reshape(X_d, [1,-1]), dtype='float64')), 
                        num_or_size_splits = 2, axis=1)
        x = tf.reshape(x, (-1, 1))
        self.noise = tf.reshape( tf.nn.softplus(x_noise), (-1,))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l, self.noise)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.math.reduce_mean(-2*tf.math.log(self.noise) + 2*self.noise)
        loss += tf.math.reduce_mean(tf.math.log(self.v))
        loss += tf.math.reduce_mean(self.L.losses)
        return loss


class Flipout(keras.Model):
    r"""
    Implements a flipout artificial neural network for the relationship x = f(X) + b + noise, 
    where X = (nlat, nlon, n_channels) with training model y = Lx + V, where L and V 
    are obtained via GP regression.
    
    Input arguments - nlon, number of rows (longitude) of X
                    - nlat, number of columns (latitude) of X
                    - n_channels, number of columns of X
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
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(Flipout, self).__init__(name='flipout_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        l1 = tf.keras.regularizers.l1(1e-3)
        # Parameters, w, b, input_noise
        # Linear weights
        num_filters = 8
        self.L = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.input_dim, self.n_features)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.input_dim, 
                    activation='relu', 
                    kernel_regularizer= l1,
                    bias_regularizer = l1),
                tf.keras.layers.Dense(self.input_dim, 
                activation='relu', 
                kernel_regularizer= l1,
                    bias_regularizer = l1),
                tfp.layers.DenseFlipout(self.input_dim, ),
            ]
        )
 
    
    def sample(self, X_d, training = True):
        eps = 1e-2*tf.random.normal(shape=(50, self.input_dim, self.n_features), dtype='float64')
        z = X_d + eps
        return self.L(z, training=training)
                
    def call(self, X_d, training=True):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        x_samp = self.sample(tf.cast(tf.reshape(X_d, [1, self.input_dim, self.n_features]), dtype='float64'), training=training)
        x = tf.reduce_mean(x_samp, axis=0)
        self.noise = tf.reshape(tf.reduce_mean( (x - x_samp)**2, axis=0), (-1,))
        x = tf.reshape(x, (-1, 1))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l, self.noise)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.reduce_mean(self.L.losses)
        loss += tf.math.reduce_mean(tf.math.log(self.v))
        loss += tf.math.reduce_mean(-2*tf.math.log(self.noise) + 2*self.noise)
        return loss


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
    def __init__(self, input_dim, n_features, x_l, latent_dim = 40, dtype='float64', **kwargs):
        super(VAE, self).__init__(name='variational_autoencoder', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        self.latent_dim = latent_dim
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l, input_noise = False)
        
        
        # Parameters, w, b, input_noise
        # Linear weights
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.n_features*self.input_dim,),
            tf.keras.layers.Dense(self.input_dim, activation='relu'),
            tf.keras.layers.Dense(self.latent_dim + self.latent_dim ),
        ])

        self.decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.latent_dim,),
            tf.keras.layers.Dense(self.input_dim, activation='relu',),
            tf.keras.layers.Dense(self.input_dim*self.n_features),
        ])

        self.regressor = tf.keras.Sequential([
            tf.keras.layers.InputLayer(self.latent_dim,),
            tf.keras.layers.Dense(self.input_dim, activation='relu',),
            tf.keras.layers.Dense(self.input_dim, activation='relu',),
            tf.keras.layers.Dense(self.input_dim),
        ])

 

    @tf.function
    def sample(self, mean, logvar, eps=None):
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
        x_samp = self.sample(self.mean, self.logvar)
        x = tf.reduce_mean(x_samp, axis=0)
        self.noise = tf.reshape(tf.reduce_mean( (x_samp - x)**2, axis=0 ), (-1,))
        x = tf.reshape(x, (-1, 1))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l, self.noise)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.math.reduce_mean((self.X_d - self.xp)**2)
        loss += tf.math.reduce_mean(tf.math.log(self.v))
        loss += tf.math.reduce_mean( self.z**2 )
        loss -= tf.math.reduce_mean( (self.z - self.mean)**2*tf.exp(-self.logvar) + self.logvar )
        return loss

class CNN(keras.Model):
    r"""
    Implements a convolutional neural network for the relationship x = f(X) + b + noise, 
    where X = (nlat, nlon, n_channels) with training model y = Lx + V, where L and V 
    are obtained via GP regression.
    
    Input arguments - nlon, number of rows (longitude) of X
                    - nlat, number of columns (latitude) of X
                    - n_channels, number of columns of X
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
    def __init__(self, nlat, nlon, n_channels, x_l, dtype='float64', **kwargs):
        super(CNN, self).__init__(name='convolutional_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.nlat = nlat
        self.nlon = nlon
        self.n_channels = n_channels
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        
        # Parameters, w, b, input_noise
        # Linear weights
        num_filters = 8
        self.L = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.nlat, self.nlon, self.n_channels)),
                tf.keras.layers.Conv2D(filters = 16, kernel_size = 9, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters = 16, kernel_size = 9, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='relu', strides=2),
                tf.keras.layers.Conv2D(filters = 32, kernel_size = 9, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters = 32, kernel_size = 9, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu', strides=2),
                tf.keras.layers.Conv2D(filters = 64, kernel_size = 9, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters = 64, kernel_size = 9, activation='relu', padding='same'),
                tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.nlat*self.nlon),
            ]
        )
 

                
    def call(self, X_d):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        x = self.L(tf.cast(tf.reshape(X_d, [1, self.nlat, self.nlon, self.n_channels]), dtype='float64'))
        x = tf.reshape(x, (-1, 1))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.math.reduce_mean(tf.math.log(self.v))
        return loss


class ResnetIdentityBlock(keras.Model):
    def __init__(self, kernel_size, filters, identity=False, name='ResnetBlock'):
        super(ResnetIdentityBlock, self).__init__(name=name, dtype='float64')
        filters1, filters2, filters3 = filters
        self.identity = identity
        self.conv2a = tf.keras.layers.Conv2D(filters1, (1,1))
        self.bn2a = tf.keras.layers.LayerNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.LayerNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1,1))
        self.bn2c = tf.keras.layers.LayerNormalization()

        self.convid = tf.keras.layers.Conv2D(filters3, (1,1))
    
    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training = training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training = training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training = training)
        if self.identity:
            x += input_tensor
        else:
            x += self.convid(input_tensor, training=training)
        return tf.nn.relu(x)

class Resnet(keras.Model):
    r"""
    Implements a residual neural network for the relationship x = f(X) + b + noise, 
    where X = (nlat, nlon, n_channels) with training model y = Lx + V, where L and V 
    are obtained via GP regression.
    
    Input arguments - nlon, number of rows (longitude) of X
                    - nlat, number of columns (latitude) of X
                    - n_channels, number of columns of X
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
    def __init__(self, nlat, nlon, n_channels, x_l, dtype='float64', **kwargs):
        super(Resnet, self).__init__(name='residual_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.nlat = nlat
        self.nlon = nlon
        self.n_channels = n_channels
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        
        # Parameters, w, b, input_noise
        # Linear weights
        self.L = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape = (self.nlat, self.nlon, self.n_channels)),
                tf.keras.layers.Conv2D(16, (9, 9), activation='relu', padding = 'same'),
                tf.keras.layers.Conv2D(16, (5, 5), strides = 2, activation='relu'),
                
                ResnetIdentityBlock(9, (16, 16, 16), name='resnet1'),
                ResnetIdentityBlock(9, (16, 16, 16), name='resnet2'),
                ResnetIdentityBlock(9, (16, 16, 16), name='resnet3'),
                tf.keras.layers.Conv2D(16, (5, 5), strides = 2, activation='relu'),
                ResnetIdentityBlock(9, (2*16, 2*16, 2*16), identity=False, name='resnet4'),
                ResnetIdentityBlock(9, (2*16, 2*16, 2*16), name='resnet5'),
                ResnetIdentityBlock(9, (2*16, 2*16, 2*16), name='resnet6'),
                tf.keras.layers.Conv2D(2*16, (5, 5), strides = 2, activation='relu'),
                ResnetIdentityBlock(9, (4*16, 4*16, 4*16), identity=False, name='resnet7'),
                ResnetIdentityBlock(9, (4*16, 4*16, 4*16), name='resnet8'),
                ResnetIdentityBlock(9, (4*16, 4*16, 4*16), name='resnet9'),
                # tf.keras.layers.Conv2D(self.nlat, (5, 5), strides = 2, activation='relu'),              
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.nlat*self.nlon, activation = 'relu',
                    kernel_regularizer = tf.keras.regularizers.l2(1e-3) ),
                tf.keras.layers.Dense(self.nlat*self.nlon,kernel_regularizer = tf.keras.regularizers.l2(1e-3) ),
            ]
        )               
    def call(self, X_d, training=True):
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        x = self.L(tf.cast(tf.reshape(X_d, [1, self.nlat, self.nlon, self.n_channels]), dtype='float64'), training)
        x = tf.reshape(x, (-1, 1))
        return x
    
    def log_prob(self, y_pred, y_l, y_true):
        self.m, self.v = self.gp(y_pred, y_l)
        loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
        loss += tf.math.reduce_mean(self.L.losses)
        loss += tf.math.reduce_mean(tf.math.log(self.v))
        return loss

####################################################################################
########################### Training Proceedure ####################################
####################################################################################
from scipy.stats import pearsonr
def train_func(dataset, model, epochs = 500, print_epoch = 100, lr = 0.001, num_early_stopping = 500, mini_batch_size=10):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    losses = np.zeros((epochs,6))
    variables = np.zeros((epochs,6))
    
    n_steps_train = dataset.i_train.size
    n_steps_test = dataset.i_test.size

    # Implement early_stopping
    early_stopping_counter = 0
    early_stopping_min_value = 1e6
    # Iterate over epochs.
    var = dataset.calc_covariance(dataset.i_train)
    for epoch in range(epochs):
        loss = 0
        batch = np.random.permutation(n_steps_train)

        for steps in range(1, n_steps_train+1):
            X_d, X_l, y_d, y_l = dataset.get_index(dataset.i_train[batch[steps-1]])
            with tf.GradientTape() as dtape:
                dtape.watch(model.trainable_variables[2:])
                x = model(X_d.reshape(-1,3) + 1e-2*np.random.randn(X_d.shape[0], 3))
                loss += model.log_prob(x, y_l.reshape(-1,2), y_d)/mini_batch_size 

            
            # Stores losses and variables per epoch for troubleshooting
            losses[epoch, 0] += tf.math.reduce_mean( (y_d-model.m)*(y_d-model.m))/n_steps_train
            losses[epoch, 1] += np.mean( (y_d < (model.m+np.sqrt(model.v) )) & 
                                          (y_d > (model.m-np.sqrt(model.v) )) )/n_steps_train
            losses[epoch, 2] += pearsonr(model.m.numpy().flatten(), y_d.flatten())[0]/n_steps_train
            variables[epoch, 0] += model.gp.amplitude.numpy()/n_steps_train
            variables[epoch, 1] += model.gp.length_scale.numpy()/n_steps_train
            variables[epoch, 2] += tf.math.reduce_mean(tf.exp(model.gp.input_noise)).numpy()/n_steps_train
            variables[epoch, 4] += np.linalg.norm(model.m.numpy())/n_steps_train
            variables[epoch, 3] += np.linalg.norm(model.v.numpy())/n_steps_train
            variables[epoch, 5] += np.linalg.norm(x.numpy())/n_steps_train

            if steps % mini_batch_size == 0:
                # Apply gradients to model parameters
                grads = dtape.gradient(loss, model.trainable_variables[2:])
                optimizer.apply_gradients(zip(grads, model.trainable_variables[2:]))
                loss = 0

        
        for steps in range(n_steps_test):
            X_d, X_l, y_d, y_l = dataset.get_index(dataset.i_test[steps])     
            x = model(X_d)
            loss = model.log_prob(x, y_l.reshape(-1,2), y_d) 
            losses[epoch, 3] += tf.math.reduce_mean( (y_d-model.m)*(y_d-model.m))/n_steps_test
            losses[epoch, 4] += np.mean( (y_d < (model.m+np.sqrt(model.v) )) & 
                                          (y_d > (model.m-np.sqrt(model.v) )) )/n_steps_test
            losses[epoch, 5] += pearsonr(model.m.numpy().flatten(), y_d.flatten())[0] / n_steps_test


        if epoch % print_epoch == 0:
            print('Epoch', epoch,  ' ', 'mean train loss = {:2.3f}'.format(losses[epoch,0]),
                                        'train Prob = {:.2f}'.format(losses[epoch,1]),
                                        'train correlation = {:.2f}'.format(losses[epoch,2]),
                                        '\n \t',
                                        'mean test loss = {:2.3f}'.format(losses[epoch, 3]), 
                                        'test Prob = {:.2f}'.format(losses[epoch,4]),
                                         'test correlation = {:.2f}'.format(losses[epoch,5]),
                 )
        
        if losses[epoch,3] < early_stopping_min_value:
            early_stopping_min_value = losses[epoch, 3]
            early_stopping_counter = 0
            model.save_weights('./saved_model/temp/temp_model_save')
        else:
            early_stopping_counter += 1
        if early_stopping_counter > num_early_stopping:
            print('Early Stopping at iteration {:d}'.format(epoch))
            break
    model.load_weights('./saved_model/temp/temp_model_save')
    model.save_weights('./saved_model/'+model.name+'{:.3f}'.format(early_stopping_min_value))
    return losses[:epoch, :], variables[:epoch, :]

####################################################################################
########################### Plotting functions #####################################
####################################################################################
def plot_stats(var, time, lats, lons, text, quantile = 0.5):
    import matplotlib.gridspec as gridspec
    # Plot the 'stat' statistic (either 'mean', or 'std') of variable 'var'
    # Along with temporal cdf's of randomly chosen points
    # Pass lat and lon values

    assert lats.size == var.shape[1], 'lat dimension does not match'
    assert lons.size == var.shape[2], 'lon dimension does not match'
    means = np.nanmean(var, axis=(1, 2))
    sort_i = np.argsort(means)
    sort_means = means[sort_i]
    q = np.quantile(means, quantile)
    print(q)
    q_i = np.argwhere(sort_means > q)[0]
    print(q_i)
    t_i = sort_i[q_i][0]
    print(t_i)

    fig = plt.figure(figsize=(18,5), constrained_layout = True)
    gs = gridspec.GridSpec(nrows=1, ncols=2, figure = fig, width_ratios = [2, 1])

    ax1 = fig.add_subplot(gs[0, :-1], projection=ccrs.Mollweide())

    cs = ax1.contourf(lons, lats, var[t_i], cmap = 'Spectral', transform=ccrs.PlateCarree())
    plt.colorbar(cs, ax = ax1, aspect=10, fraction = 0.05)
    ax1.coastlines()
    ax1.set_aspect('auto', adjustable=None)
    tt = pd.to_datetime(str(time[t_i])).strftime('%Y.%m.%d')
    ax1.set_title(str(quantile) + ' Quantile of ' + text + '\n time :' + tt )

    ax2 = fig.add_subplot(gs[0, -1])
    i = np.random.choice(np.arange(10, lats.size-10), 80)
    j = np.random.choice(np.arange(10, lons.size-10), 80)
    for ik, jk in zip(i, j):
        if np.isnan(var[0, ik, jk]):
            continue
        values, base = np.histogram(var[:, ik, jk], bins=40)
        cumulative = np.cumsum(values)/var.shape[0]
        ax2.plot(base[:-1], cumulative, c='blue', linewidth=0.2)
        ax1.scatter(lons[jk], lats[ik], c= 'k', s = 80, marker='*', transform=ccrs.PlateCarree())
    values, base = np.histogram(np.random.randn(var.shape[0]), bins = 40)
    cumulative = np.cumsum(values)/var.shape[0]
    ax2.plot(base[:-1], cumulative, 'k--', linewidth=2.0, label='Normal CDF')
    ax2.set_title('Temporal CDF of Starred Locations')
    ax2.grid()
    ax2.legend()
    ax2.set_xlim(-3, 3)
    plt.savefig(text+'_' + str(int(100*quantile)) + '_' + 'quantile.png')
    plt.show()
    
def plot_eof(var, lats, lons, text, n = 2):
    from sklearn.decomposition import PCA
    import matplotlib.gridspec as gridspec
    # Plot leading 'n' EOF's of variable 'var' in one of the two
    # following axes set up,   limit n to 5      
    # if n is even:                if n is odd:
    #    EOF1 EOF2                  EOF1 EOF2
    #      ...                        ...
    #    EOF-2 EOF-1                EOF-1 sing. values
    #     sing. values
    n = min(n, 5)

    pca = PCA(n_components=15)
    mask = np.isfinite(var.values.mean(axis=0))
    pca.fit(var.values[:, mask])
    v = np.NaN*np.ones_like(var[0])

    fig = plt.figure(figsize = (16,4))
    
    gs = gridspec.GridSpec(nrows=n//2, ncols=3, figure = fig, width_ratios = [10, 10, 1])
    levs = np.linspace(np.nanmin(pca.components_[:n]), np.nanmax(pca.components_[:n]), 15)
    for j in range(n):
        row = j // 2
        col = j % 2
        axes = fig.add_subplot(gs[row, col], projection=ccrs.Robinson())
        v[mask] = pca.components_[j]
        cs = axes.contourf(lons, lats, v, levels = levs, cmap='Spectral', transform=ccrs.PlateCarree())
#         plt.colorbar(cs, ax = axes)
        axes.coastlines()
        axes.set_title('EOF # ' + str(j+1) + ' with explained variance ratio = {:1.2f}'.format(pca.explained_variance_ratio_[j]))
        axes.set_aspect('auto', adjustable=None)
    
    f3_ax4 = fig.add_subplot(gs[:, -1])
    plt.colorbar(cs, cax = f3_ax4, aspect = 5, fraction = 0.01);
    plt.suptitle(text)
    plt.savefig(text+'_'+str(n)+'_'+'EOF.png')
    plt.show()

def plot_hovmoller(var, time, lats, lons, string):
    import matplotlib.gridspec as gridspec
    import metpy.calc as mpcalc
    import cartopy.feature as cfeature
    pd.plotting.register_matplotlib_converters()
    # Start figure
    ## Make Hovmuller Diagram

    # Compute weights and take weighted average over latitude dimension
    weights = np.cos(np.deg2rad(lats))
    avg_data = (var * weights[None, :, None]).sum(axis=1) / np.sum(weights)

    # Get times and make array of datetime objects
    vtimes = time.astype('datetime64[ms]').astype('O')

    fig = plt.figure(figsize=(15, 20))

    # Use gridspec to help size elements of plot; small top plot and big bottom plot
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1, 6], hspace=0.03)

    # Tick labels
    x_tick_labels = [u'180\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}W',
                     u'0\N{DEGREE SIGN}E', u'90\N{DEGREE SIGN}E',
                     u'180\N{DEGREE SIGN}E']

    # Top plot for geographic reference (makes small map)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1.set_extent([-180, 180, np.round(lats.min())-5, np.round(lats.max())+5], ccrs.PlateCarree())
    ax1.set_yticks([np.round(lats.min()), np.round(lats.max())])
    ax1.set_yticklabels([u'{:2.1f}\N{DEGREE SIGN}S'.format(np.round(lats.min())), u'{:2.1f}\N{DEGREE SIGN}S'.format(np.round(lats.max()))])
    ax1.set_xticks([-180, -90, 0, 90, 180])
    ax1.set_xticklabels(x_tick_labels)
    ax1.grid(linestyle='dotted', linewidth=2)

    # Add geopolitical boundaries for map reference
    ax1.add_feature(cfeature.COASTLINE.with_scale('50m'))
    ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

    # Set some titles
    plt.title('Hovmoller Diagram', loc='left')
    plt.title(string, loc='right')

    # Bottom plot for Hovmoller diagram
    ax2 = fig.add_subplot(gs[1, 0])
    # ax2.invert_yaxis()  # Reverse the time order to do oldest first

    # Plot of chosen variable averaged over latitude and slightly smoothed
    # clevs = np.arange(-5, 5, 10)
    cf = ax2.contourf(lons, vtimes, mpcalc.smooth_n_point(
        avg_data, 9, 2), cmap=plt.cm.bwr, extend='both')
    cs = ax2.contour(lons, vtimes, mpcalc.smooth_n_point(
        avg_data, 9, 2), colors='k', linewidths=1)
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)
    cbar.set_label('m $s^{-1}$')

    # Make some ticks and tick labels
    ax2.set_xticks([-177.5, -90, 0, 90, 177.5])
    ax2.set_xticklabels(x_tick_labels)
    ax2.set_yticks(pd.date_range(start=vtimes[0], end=vtimes[-1], freq="3MS").strftime('%Y.%m.%d'))
    ax2.set_yticklabels(pd.date_range(start=vtimes[0], end=vtimes[-1], freq="3MS").strftime('%Y.%m.%d'))

    # Set some titles
    plt.title(string, loc='left', fontsize=10)
    plt.title('Time Range: {0:%Y%m} - {1:%Y%m}'.format(vtimes[0], vtimes[-1]),
              loc='right', fontsize=10)
    
    if lats[0] < 0:
        min_lat_string = str(int(-lats[0]))+'S'
    else:
        min_lat_string = str(int(lats[0]))+'N'
    if lats[-1] <0:
        max_lat_string = str(int(-lats[-1]))+'S'
    else:
        max_lat_string = str(int(lats[-1]))+'N'
    plt.savefig(string+'_'+min_lat_string+'_'+max_lat_string+'_'+'hovmoller.png')

    plt.show()
