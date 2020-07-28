####################################################################################
####################################################################################
########################### File Description #######################################
####################################################################################
####################################################################################

# This file creates a class that imports preprocessed sss data, sst data, and ssh data 
# as well as mld data using the dataset class and methods. Using this class, users 
# can treat the 'get_index' method as an iterator to draw individual weeks from the
# data set for use in training various machine learning models.

# There are a number of machine learning models coded in this documents. The first
# two class set up the gaussian process regression:
#           1. ExponentiatedQuadratic - implements kernel with haversine distance
#           2. GPR - implements gaussian process regression. Optimizes hyper parameters
#                   by maximum likelihood 
#           
#  Next machine learning models take sss, sst, ssh data as inputs and produce dense
#  grid predictions. Using the log_prob method of each model class we link the dense
#  grid to the sparse mld argo observation network via the gaussian process regression
#  class. The available models are:
#           3. Linear
#           4. ANN
#           5. Variational ANN (ANN_distribution)
#           6. Dropout ANN
#           7. Bayesian NN (Flipout)
#           8. Variational AutoEncoder (VAE)
#  For full details, see associated text with each method. 
# 
# train_minibatch is the currently implemented training routine. As the title suggests,
# It implements a minibatch training algorithm, currently using Adam with a user-specified
# learning rate. It estimates log loss, probabilistic calibration, and correlation 
# coefficient for the training and testing set, which is then printed. It has a built in
# early stopping criterion currently based off of progress in the test correlation coefficient.
#
#
# An minimal working example:
# import numpy as np
# import models
# import tensorflow as tf
#
# # Select a certain lat, lon region
# lat_bounds = np.array([-25, -5])
# lon_bounds = np.array([65, 85])
#
# # Create the dataset
# data = models.dataset(lat_bounds, lon_bounds)
# 
# # Normalize data according to training data
# data.normalize()
# 
# # Obtain the indices for the train/test/val split (determined during preprocessing)
# i_train, i_test, i_val = data.i_train, data.i_test, data.i_val
#
# # Usage for obtaining data from dataset
# # .get_index( index ) retrieves the data according to week corresponding to index
# # X_d - input data (sss, sst, sha)
# # X_l - input data locations (lon, lat)
# # y_d - mld data
# # y_l - mld locations
#
# X_d, X_l, y_d, y_l = data.get_index(i_train[0])
#
# # Call of methods is models.METHOD( input dimensions, number of features, input data locations )
# input_dim = X_d.shape[0]
# n_features = X_d.shape[1]
# lp = models.Linear(input_dim, n_features, X_l)
# 
# # Train model
# loss  = models.train_func(data,                       # Pass the data set
#                           lp,                         # Pass the model
#                           epochs=300,                 # Max Number of Epochs
#                           print_epoch = 1,            # Printing frequency
#                           lr = 3e-4,                  # Optimizer learning rate
#                           num_early_stopping = 20,    # Epochs from optimum to continue optimizer
#                           mini_batch_size = 25,       # Batch size should be divisor of 150
#                           )





####################################################################################
########################### Import libraries #######################################
####################################################################################
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import cartopy
import cartopy.crs as ccrs
from mpl_toolkits.basemap import maskoceans
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions
tfb = tfp.bijectors

####################################################################################
########################### Import Data ############################################
####################################################################################

            
class dataset:
    def __init__(self, lat_bounds, lon_bounds, anomalies=True, CNN = False):

        with xr.open_dataset('./data/sss_sst_ssh_normed_anomalies_weekly.nc') as ds:
            print(ds)
            sal = ds.salinity.values.astype(np.float64)
            sal_anom = ds.salinity_anomaly.values.astype(np.float64)
            temp = ds.temperature.values.astype(np.float64)
            temp_anom = ds.temperature_anomaly.values.astype(np.float64)
            height = ds.height.values.astype(np.float64)
            lat = ds.lat.values.astype(np.float64)
            lon = ds.lon.values.astype(np.float64)
            self.time = ds.time.values
            self.i_train = ds.training_index.values
            self.i_test = ds.testing_index.values
            self.i_val  = ds.validation_index.values

        with xr.open_dataset('./data/mldbmax_full_anomalies_weekly_full.nc') as ds:
            print(ds)
            mldb = ds.copy()
            self.y_week = mldb['week']

        if anomalies:
            self.X_data = np.stack((sal_anom, temp_anom, height), axis=-1)
            self.y_data = mldb['mldb']
        else:
            self.X_data = np.stack((sal, temp, height), axis=-1)
            self.y_data = mldb['mldb_full']


        lat_mask = (lat < lat_bounds[1]) & (lat > lat_bounds[0])
        lon_mask = (lon < lon_bounds[1]) & (lon > lon_bounds[0])
        self.mask = np.isfinite(self.X_data[:, :, :, 0].mean(axis=0))
        if CNN:
            self.X_data[:, ~self.mask, :] = 0.0
            self.mask = np.isfinite(self.X_data[:, :, :, 0].mean(axis=0))
        self.mask[~lat_mask, :] = False
        self.mask[:, ~lon_mask] = False
        self.X_data = self.X_data[:, self.mask, :]
        
        [LAT, LON] = np.meshgrid(lat, lon, indexing='ij')
        self.X_loc = np.stack((LON[self.mask], LAT[self.mask]), axis=-1)

        self.y_data = self.y_data.where(
            (mldb.lat < lat_bounds[1]) & (mldb.lat > lat_bounds[0]) & \
            (mldb.lon < lon_bounds[1]) & (mldb.lon > lon_bounds[0])
            ).dropna('index')
        self.y_week = self.y_week.where(
            (mldb.lat < lat_bounds[1]) & (mldb.lat > lat_bounds[0]) & \
            (mldb.lon < lon_bounds[1]) & (mldb.lon > lon_bounds[0])
            ).dropna('index')
        self.y_lat = mldb['lat'].where(
            (mldb.lat < lat_bounds[1]) & (mldb.lat > lat_bounds[0]) & \
            (mldb.lon < lon_bounds[1]) & (mldb.lon > lon_bounds[0])
            ).dropna('index')
        self.y_lon = mldb['lon'].where(
            (mldb.lat < lat_bounds[1]) & (mldb.lat > lat_bounds[0]) & \
            (mldb.lon < lon_bounds[1]) & (mldb.lon > lon_bounds[0])
            ).dropna('index')

        self.lat_bounds = lat_bounds
        self.lon_bounds = lon_bounds
        
    def normalize(self):
        
        for i in range(self.X_data.shape[-1]):
            self.X_data[:, :, i] = normalize(self.X_data[:, :, i], self.i_train)
        
        self.y_mean = self.y_data.where(np.in1d(self.y_week, self.time[self.i_train])).dropna('index').mean().values
        self.y_std = self.y_data.where(np.in1d(self.y_week, self.time[self.i_train])).dropna('index').std().values
        self.y_data = (self.y_data - self.y_mean)/self.y_std
        
    def get_index(self, index):
        y = self.y_data.where(np.in1d(self.y_week, self.time[index])).dropna('index').values
        y_loc = np.stack( (self.y_lon.where(np.in1d(self.y_week, self.time[index])).dropna('index'), 
                            self.y_lat.where(np.in1d(self.y_week, self.time[index])).dropna('index')),
                            axis=-1)
        return self.X_data[index].astype(np.float64), self.X_loc.astype(np.float64), y.astype(np.float64), y_loc.astype(np.float64)
    

####################################################################################
########################### Preprocess Models ######################################
####################################################################################

def normalize(inputs, train_index):
    return (inputs - np.nanmean(inputs[train_index], axis=0))/np.nanstd(inputs[train_index], axis=0)



####################################################################################
########################### Define Models ##########################################
####################################################################################

from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import PositiveSemidefiniteKernel

class ExponentiatedQuadratic(PositiveSemidefiniteKernel):
    def __init__(self,
               amplitude=None,
               length_scale=None,
               feature_ndims=1,
               validate_args=False,
               name='ExponentiatedQuadratic'):
        parameters = dict(locals())
        with tf.name_scope(name):
            dtype = util.maybe_get_common_dtype(
                [amplitude, length_scale])
        self._amplitude = tensor_util.convert_nonref_to_tensor(
          amplitude, name='amplitude', dtype=dtype)
        self._length_scale = tensor_util.convert_nonref_to_tensor(
          length_scale, name='length_scale', dtype=dtype)
        super(ExponentiatedQuadratic, self).__init__(
              feature_ndims,
              dtype=dtype,
              name=name,
              validate_args=validate_args,
              parameters=parameters)

    @property
    def amplitude(self):
        """Amplitude parameter."""
        return self._amplitude

    @property
    def length_scale(self):
        """Length scale parameter."""
        return self._length_scale

    def _batch_shape(self):
        scalar_shape = tf.TensorShape([])
        return tf.broadcast_static_shape(
            scalar_shape if self.amplitude is None else self.amplitude.shape,
            scalar_shape if self.length_scale is None else self.length_scale.shape)

    def _batch_shape_tensor(self):
        return tf.broadcast_dynamic_shape(
            [] if self.amplitude is None else tf.shape(self.amplitude),
            [] if self.length_scale is None else tf.shape(self.length_scale))

    def _apply(self, x1, x2, example_ndims=0):
        x1 = tf.reshape(x1, (-1, 2))
        x2 = tf.reshape(x2, (-1, 2))
        exponent = -0.5 * haversine_dist(x1, x2)
        if self.length_scale is not None:
            length_scale = tf.convert_to_tensor(self.length_scale)
            length_scale = util.pad_shape_with_ones(
              length_scale, example_ndims)
            exponent = exponent / length_scale**2

        if self.amplitude is not None:
            amplitude = tf.convert_to_tensor(self.amplitude)
            amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
            exponent = exponent + 2. * tf.math.log(amplitude)
        return tf.exp(exponent)

    def _parameter_control_dependencies(self, is_init):
        if not self.validate_args:
            return []
        assertions = []
        for arg_name, arg in dict(amplitude=self.amplitude,
                              length_scale=self.length_scale).items():
            if arg is not None and is_init != tensor_util.is_ref(arg):
                assertions.append(assert_util.assert_positive(
                arg,
                message='{} must be positive.'.format(arg_name)))
        return assertions

def haversine_dist(X, X2):
    pi = np.pi / 180
    f = tf.expand_dims(X * pi, -2)  # ... x N x 1 x D
    f2 = tf.expand_dims(X2 * pi, -3)  # ... x 1 x M x D
    d = tf.sin((f - f2) / 2) ** 2
    lat1, lat2 = tf.expand_dims(X[:, 0] * pi, -1), \
                tf.expand_dims(X2[:, 0] * pi, -2)
    cos_prod = tf.cos(lat2) * tf.cos(lat1)
    a = d[:,:,0] + cos_prod * d[:,:,1]
    c = tf.asin(tf.sqrt(a)) * 6371 * 2
    return c

class GPR(keras.Model):
    def __init__(self, x_l, dtype='float64', **kwargs):
        super(GPR, self).__init__(name='gaussian_process', dtype='float64', **kwargs)
        self.x_l = tf.cast(x_l, dtype='float64')
        self.input_dim = self.x_l.shape[0]

        self.amplitude = tf.Variable(
            initial_value=1.,
            constraint = tf.keras.constraints.NonNeg(),
            name='amplitude',
            trainable=True,
            dtype=np.float64)

        self.length_scale = tf.Variable(
            initial_value=20.,
            constraint = tf.keras.constraints.NonNeg(),
            name='length_scale',
            trainable=True,
            dtype=np.float64)

        self.noise_variance = tf.Variable(
            initial_value=1.e-3,
            constraint = tf.keras.constraints.NonNeg(),
            trainable=True,
            name='observation_noise_variance_var',
            dtype=np.float64)
        

    def optimize(self, x, gp, optimizer):
        with tf.GradientTape() as tape:
            tape.watch(gp.trainable_variables)
            loss = -gp.log_prob(x)
        grads = tape.gradient(loss, gp.trainable_variables)
        optimizer.apply_gradients(zip(grads, gp.trainable_variables))
        return loss
    
    def call(self, x, y_l, noise = None):
        # Calculuates a gaussian process
        # m = Lx + V
        # where L = Kyx (Kxx + \sigmaI)^{-1}
        # where V = Kyy - L Kxy + L\sigmaIL^{T}
        kernel = ExponentiatedQuadratic(
                    self.amplitude,
                    self.length_scale)
        # gp = tfd.GaussianProcess(
        #     kernel=kernel,
        #     index_points=self.x_l,
        #     observation_noise_variance=self.noise_variance,
        #     jitter = 1e-6)
        # optimizer = tf.optimizers.Adam(learning_rate=1e-2)
        # for i in range(10):
            # neg_log_likelihood_ = self.optimize(x, gp, optimizer)

                
        Kxx = kernel._apply(self.x_l, self.x_l) + self.noise_variance*tf.eye(self.input_dim, dtype='float64')
        if noise is not None:
            Kxx += tf.linalg.diag(noise)
        Kxy = kernel._apply(self.x_l, y_l)
        Kyy = kernel._apply(y_l, y_l)
        Kyy = tf.linalg.set_diag( Kyy, tf.linalg.diag_part(Kyy) + 1e-6)
        K_chol = tf.linalg.cholesky(Kxx)
        self.m = tf.linalg.matmul(Kxy,
                            tf.linalg.cholesky_solve(K_chol, tf.reshape(x, (-1,1))),
                            transpose_a = True)
        self.V = Kyy - tf.linalg.matmul(Kxy,
                                  tf.linalg.cholesky_solve(K_chol, Kxy),
                                  transpose_a = True)
        self.V_chol = tf.linalg.cholesky(self.V)
        return tf.reshape(self.m, (-1,)), tf.linalg.diag_part(self.V)
    
    def sample(self, size, num_samples = 1):
        noise = tf.random.normal( (size, num_samples), dtype='float64')
        sample = self.m + tf.linalg.matmul(
                    self.V_chol,
                    noise,
                    )
        return tf.reshape(sample, (-1,1))
    
    def log_prob(self, x):
        z = (self.m - tf.reshape(x, (-1,1)))
        l = -0.5 * tf.matmul(z, 
                            tf.linalg.cholesky_solve(self.V_chol,
                                         z),
                            transpose_a = True)
        l -= 0.5*tf.math.reduce_sum(tf.math.log(tf.linalg.diag_part(self.V_chol)))
        return l
    
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
        x = tf.reshape(x, (-1,))
        
        return x
    
    
    def log_prob(self, y_pred, y_l, y_true, y_size):
        self.m, self.v = self.gp(y_pred, y_l)
        self.sample = self.gp.sample(y_size)
        loss = -self.gp.log_prob( y_true )
        return loss
    
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
        
        l1 = tf.keras.regularizers.l2(5e-3)
        
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
        x = tf.reshape(x, (-1, ))
        return x
    
    def log_prob(self, y_pred, y_l, y_true, y_size):
        self.m, self.v = self.gp(y_pred, y_l)
        self.sample = self.gp.sample(y_size)
        loss = -self.gp.log_prob( y_true )
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
        
        l1 = tf.keras.regularizers.l2(1e-4)
        
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
            tf.keras.layers.Dropout(0.3)
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
        r"""
        Produces an estimate x for the latent variable x = f(X) + noise
        With that estimate x, projects to the output space m = Lx + var
        where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
        outputs x, mean and var
        """
        
        ## ANN Map
        x = tf.reshape(self.L(tf.cast(tf.reshape(X_d, [1,-1]), dtype='float64'), training=True), (-1, ))
        return x
    
    def log_prob(self, y_pred, y_l, y_true, y_size):
        self.m, self.v = self.gp(y_pred, y_l)
        self.sample = self.gp.sample(y_size)
        loss = -self.gp.log_prob( y_true )
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
        super(ANN_distribution, self).__init__(name='variational_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.input_dim = input_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        l1 = tf.keras.regularizers.l1(5e-2)
        l2 = tf.keras.regularizers.l2(5e-3)
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
        x = self.x_dist.sample()

        x = tf.reshape(x, (-1,))
        self.noise = tf.reshape( self.x_dist.variance(), (-1,))
        return x
    
    def log_prob(self, y_pred, y_l, y_true, y_size):
        self.m, self.v = self.gp(y_pred, y_l)
        self.sample = self.gp.sample(y_size)
        loss = -self.gp.log_prob( y_true )
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
 
    
    def funcsample(self, X_d, training = True):
        eps = 1e-3*tf.random.normal(shape=(50, self.input_dim, self.n_features), dtype='float64')
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
        x_samp = self.funcsample(tf.cast(tf.reshape(X_d, 
                                                    [1, self.input_dim, self.n_features]), 
                                    dtype='float64'), 
                                 training=training)
        x = tf.reduce_mean(x_samp, axis=0)
        self.noise = tf.reduce_mean( (x - x_samp)**2, axis=0)
        x = tf.reshape(x, (-1, ))
        return x
    
    def log_prob(self, y_pred, y_l, y_true, y_size):
        self.m, self.v = self.gp(y_pred, y_l)
        self.sample = self.gp.sample(y_size)
        loss = -self.gp.log_prob( y_true )
        loss += tf.math.reduce_mean(self.L.losses)
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
        self.gp = GPR(x_l)
        
        
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
    
    def log_prob(self, y_pred, y_l, y_true, y_size):
        self.m, self.v = self.gp(y_pred, y_l)
        self.sample = self.gp.sample(y_size)
        loss = -self.gp.log_prob( y_true )
        loss += tf.math.reduce_mean((self.X_d - self.xp)**2)
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
    def __init__(self, nlat, nlon, n_features, x_l, dtype='float64', **kwargs):
        super(CNN, self).__init__(name='convolutional_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.nlat = nlat
        self.nlon = nlon
        self.n_features = n_features
        self.input_dim = nlat*nlon
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = GPR(x_l)
        
        l1 = tf.keras.regularizers.l1(1e-2)
        # Parameters, w, b, input_noise
        # Linear weights
        num_filters = 8
        self.L = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.nlat, self.nlon, self.n_features)),
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
                tf.keras.layers.Dense(self.input_dim, 
                                       ),
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
        x = self.L(tf.cast(tf.reshape(X_d, [1, self.nlat, self.nlon, self.n_features]), dtype='float64'))
        x = tf.reshape(x, (-1, ))
        return x
    
    def log_prob(self, y_pred, y_l, y_true, y_size):
        self.m, self.v = self.gp(y_pred, y_l)
        self.sample = self.gp.sample(y_size)
        loss = -self.gp.log_prob( y_true )
        # loss += tf.math.reduce_mean(self.L.losses)
        return loss


# class ResnetIdentityBlock(keras.Model):
#     def __init__(self, kernel_size, filters, identity=False, name='ResnetBlock'):
#         super(ResnetIdentityBlock, self).__init__(name=name, dtype='float64')
#         filters1, filters2, filters3 = filters
#         self.identity = identity
#         self.conv2a = tf.keras.layers.Conv2D(filters1, (1,1))
#         self.bn2a = tf.keras.layers.LayerNormalization()

#         self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
#         self.bn2b = tf.keras.layers.LayerNormalization()

#         self.conv2c = tf.keras.layers.Conv2D(filters3, (1,1))
#         self.bn2c = tf.keras.layers.LayerNormalization()

#         self.convid = tf.keras.layers.Conv2D(filters3, (1,1))
    
#     def call(self, input_tensor, training=False):
#         x = self.conv2a(input_tensor)
#         x = self.bn2a(x, training = training)
#         x = tf.nn.relu(x)

#         x = self.conv2b(x)
#         x = self.bn2b(x, training = training)
#         x = tf.nn.relu(x)

#         x = self.conv2c(x)
#         x = self.bn2c(x, training = training)
#         if self.identity:
#             x += input_tensor
#         else:
#             x += self.convid(input_tensor, training=training)
#         return tf.nn.relu(x)

# class Resnet(keras.Model):
#     r"""
#     Implements a residual neural network for the relationship x = f(X) + b + noise, 
#     where X = (nlat, nlon, n_channels) with training model y = Lx + V, where L and V 
#     are obtained via GP regression.
    
#     Input arguments - nlon, number of rows (longitude) of X
#                     - nlat, number of columns (latitude) of X
#                     - n_channels, number of columns of X
#                     - x_l,  locations of inputs, shape (input_dim, 2)
#                             columns house (lon,lat) coordinates respectively.
#                     - y_l, locations of training data
#                     - y_d, training data values
                            
#     Output arguments - x, estimate of x = f(X) + noise
#                      - m, mean estimate of m = Lx + V
#                      - v, diagonal variance of gp covariance V
                     
#     Parameters       - w_i, linear weight matrix, shape (input_dim, n_features)
#                      - b_i, bias, shape (input_dim, )
    
#     Inherited Parameters - input_noise, input-dependent noise estimate, shape (input_dim,)
#                          gives estimate of variances 
#                          - .gp.amplitude, kernel amplitude
#                          - .gp.length_Scale, kernel length scale
    
    
#     """
#     def __init__(self, nlat, nlon, n_channels, x_l, dtype='float64', **kwargs):
#         super(Resnet, self).__init__(name='residual_neural_network', dtype='float64', **kwargs)
        
#         # Sizes
#         self.nlat = nlat
#         self.nlon = nlon
#         self.n_channels = n_channels
        
#         # Initialize grid and gaussian process        
#         self.x_l = x_l
#         self.gp = GPR(x_l)
        
        
#         # Parameters, w, b, input_noise
#         # Linear weights
#         self.L = tf.keras.Sequential(
#             [
#                 tf.keras.layers.InputLayer(input_shape = (self.nlat, self.nlon, self.n_channels)),
#                 tf.keras.layers.Conv2D(16, (9, 9), activation='relu', padding = 'same'),
#                 tf.keras.layers.Conv2D(16, (5, 5), strides = 2, activation='relu'),
                
#                 ResnetIdentityBlock(9, (16, 16, 16), name='resnet1'),
#                 ResnetIdentityBlock(9, (16, 16, 16), name='resnet2'),
#                 ResnetIdentityBlock(9, (16, 16, 16), name='resnet3'),
#                 tf.keras.layers.Conv2D(16, (5, 5), strides = 2, activation='relu'),
#                 ResnetIdentityBlock(9, (2*16, 2*16, 2*16), identity=False, name='resnet4'),
#                 ResnetIdentityBlock(9, (2*16, 2*16, 2*16), name='resnet5'),
#                 ResnetIdentityBlock(9, (2*16, 2*16, 2*16), name='resnet6'),
#                 tf.keras.layers.Conv2D(2*16, (5, 5), strides = 2, activation='relu'),
#                 ResnetIdentityBlock(9, (4*16, 4*16, 4*16), identity=False, name='resnet7'),
#                 ResnetIdentityBlock(9, (4*16, 4*16, 4*16), name='resnet8'),
#                 ResnetIdentityBlock(9, (4*16, 4*16, 4*16), name='resnet9'),
#                 # tf.keras.layers.Conv2D(self.nlat, (5, 5), strides = 2, activation='relu'),              
#                 tf.keras.layers.Flatten(),
#                 tf.keras.layers.Dense(self.nlat*self.nlon, activation = 'relu',
#                     kernel_regularizer = tf.keras.regularizers.l2(1e-3) ),
#                 tf.keras.layers.Dense(self.nlat*self.nlon,kernel_regularizer = tf.keras.regularizers.l2(1e-3) ),
#             ]
#         )               
#     def call(self, X_d, training=True):
#         r"""
#         Produces an estimate x for the latent variable x = f(X) + noise
#         With that estimate x, projects to the output space m = Lx + var
#         where the loss is calculated as l = (y_d - mean)(y_d - mean)/var
#         outputs x, mean and var
#         """
        
#         ## ANN Map
#         x = self.L(tf.cast(tf.reshape(X_d, [1, self.nlat, self.nlon, self.n_channels]), dtype='float64'), training)
#         x = tf.reshape(x, (-1, 1))
#         return x
    
#     def log_prob(self, y_pred, y_l, y_true):
#         self.m, self.v = self.gp(y_pred, y_l)
#         loss = tf.math.reduce_mean((y_true-self.m)**2/self.v)
#         loss += tf.math.reduce_mean(self.L.losses)
#         loss += tf.math.reduce_mean(tf.math.log(self.v))
#         return loss

####################################################################################
########################### Training Proceedure ####################################
####################################################################################
from scipy.stats import pearsonr
@tf.function(autograph=False)
def batch_train(optimizer, model, X_d_batch, y_d_batch, y_l_batch, y_size, mini_batch_size, n_steps_train):
    loss = 0.0
    for j in range(mini_batch_size):
        with tf.GradientTape() as dtape:
            X_d = X_d_batch[j]
            y_d = y_d_batch[j]
            y_l = y_l_batch[j]
            dtape.watch(model.trainable_variables)
            x = model(X_d + tf.random.normal( (X_d.shape[0], 3), stddev=1e-4, dtype=tf.float64) )
            loss += mini_batch_size*model.log_prob(x, y_l, y_d, y_size[j])/n_steps_train
            
    # Apply gradients to model parameters
    grads = dtape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

def train_minibatch(dataset, model, mini_batch_size, optimizer):
    n_steps_train = dataset.i_train.size
    batch_index = np.random.permutation(n_steps_train)
    num_minibatch = np.ceil(n_steps_train / mini_batch_size).astype(int)
    i = 0
    for batch in range(num_minibatch):
        X_d_batch = np.zeros((mini_batch_size, model.input_dim, model.n_features))
        y_d_batch = []
        y_l_batch1 = []
        y_l_batch2 = []
        y_size = np.zeros(mini_batch_size, dtype=np.int64)
        for j in range(mini_batch_size):
            X_d, X_l, y_d, y_l = dataset.get_index(dataset.i_train[batch_index[i]])
            X_d_batch[j] = X_d
            y_d_batch.append(y_d.reshape(-1,))
            y_l_batch1.append(y_l[:,0])
            y_l_batch2.append(y_l[:,1])
            y_size[j] = y_d.size
            i+=1
        X_d_batch = tf.cast(X_d_batch, dtype='float64')
        y_d_batch = tf.ragged.constant(y_d_batch, ragged_rank=1)
        y_l_batch = tf.stack((tf.ragged.constant(y_l_batch1, ragged_rank=1),
                            tf.ragged.constant(y_l_batch2, ragged_rank=1)), axis=-1)
        batch_train(optimizer, model, X_d_batch, y_d_batch, y_l_batch, y_size, mini_batch_size, n_steps_train)
            
    

def train_func(dataset, model, epochs = 500, print_epoch = 100, lr = 0.001, num_early_stopping = 500, mini_batch_size=10):
    losses = np.zeros((epochs,7))
    
    n_steps_train = dataset.i_train.size
    n_steps_test = dataset.i_test.size

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Implement early_stopping
    early_stopping_counter = 0
    early_stopping_min_value = 0
    # Iterate over epochs.
    for epoch in range(epochs):
        
        losses[epoch, 0] = train_minibatch(dataset, model, mini_batch_size, optimizer)
        
        for steps in np.random.choice(dataset.i_train, size = 20, replace = False):
            X_d, X_l, y_d, y_l = dataset.get_index(steps)     
            x = model(X_d)
            loss = model.log_prob(x, y_l.reshape(-1,2), y_d, y_d.size) 
            losses[epoch, 1] += tf.math.reduce_mean( (y_d-model.sample)**2)/20.0
            losses[epoch, 2] += np.mean( (y_d < (model.m+np.sqrt(model.v) )) & 
                                          (y_d > (model.m-np.sqrt(model.v) )) )/20.0
            losses[epoch, 3] += pearsonr(model.sample.numpy().flatten(), y_d.flatten())[0] / 20.0
                
        for steps in range(n_steps_test):
            X_d, X_l, y_d, y_l = dataset.get_index(dataset.i_test[steps])     
            x = model(X_d)
            loss = model.log_prob(x, y_l.reshape(-1,2), y_d, y_d.size) 
            losses[epoch, 4] += tf.math.reduce_mean( (y_d-model.sample)**2)/n_steps_test
            losses[epoch, 5] += np.mean( (y_d < (model.m+np.sqrt(model.v) )) & 
                                          (y_d > (model.m-np.sqrt(model.v) )) )/n_steps_test
            losses[epoch, 6] += pearsonr(model.sample.numpy().flatten(), y_d.flatten())[0] / n_steps_test


        if epoch % print_epoch == 0:
            print('Epoch', epoch,  ' ', 'mean train loss = {:2.3f}'.format(losses[epoch,1]),
                                        'train Prob = {:.2f}'.format(losses[epoch,2]),
                                        'train correlation = {:.2f}'.format(losses[epoch,3]),
                                        '\n \t',
                                        'mean test loss = {:2.3f}'.format(losses[epoch, 4]), 
                                        'test Prob = {:.2f}'.format(losses[epoch,5]),
                                         'test correlation = {:.2f}'.format(losses[epoch,6]),
                 )
        
        if losses[epoch,6] > early_stopping_min_value:
            early_stopping_min_value = losses[epoch, 6]
            early_stopping_counter = 0
            model.save_weights('./saved_model/temp/temp_model_save')
        else:
            early_stopping_counter += 1
        if early_stopping_counter > num_early_stopping:
            print('Early Stopping at iteration {:d}'.format(epoch))
            break
    model.load_weights('./saved_model/temp/temp_model_save')
    model.save_weights('./saved_model/'+model.name+'{:.3f}'.format(early_stopping_min_value))
    return losses[:epoch, :]

def hmc(dataset, model, epochs = 500, print_epoch = 100, lr = 0.001, num_early_stopping = 500, mini_batch_size=10):
    losses = np.zeros((epochs,7))
    
    n_steps_train = dataset.i_train.size
    n_steps_test = dataset.i_test.size

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # Implement early_stopping
    early_stopping_counter = 0
    early_stopping_min_value = 0
    # Iterate over epochs.
    L = 10
    for epoch in range(epochs):
        q = model.trainable_variables.copy()
        for j in range(len(q)):
            q[j] = tf.identity(model.trainable_variables[j])
        current_U = U(dataset, model)[0]/model.input_dim
        print('Current U', current_U)
        p = model.trainable_variables.copy()
        current_P = 0.0
        for j in range(len(p)):
            p[j] = tf.random.normal(p[j].shape, stddev = 1e-4, dtype='float64')
            current_P += tf.math.reduce_sum(p[j]**2)/2.0
        print('Current P', current_P)

        grads = U(dataset, model)[1]
        for j in range(len(p)):
            p[j] = p[j] - lr * grads[j] / 2.0
        for i in range(L):
            for j in range(len(p)):
                model.trainable_variables[j].assign(
                    model.trainable_variables[j] + lr * p[j] 
                )

            if i != (L-1):
                grads = U(dataset, model)[1]
                for j in range(len(p)):
                    p[j] = p[j] - lr * grads[j]
        
        grads = U(dataset, model)[1]
        proposed_P = 0.0
        for j in range(len(p)):
            p[j] = p[j] - lr * grads[j] / 2.0
            p[j] = -p[j]
            proposed_P += tf.math.reduce_sum(p[j]**2)/2.0

        proposed_U = U(dataset, model)[0]/model.input_dim
        print('Proposed U', proposed_U)
        print('Proposed P', proposed_P)
        delta = (current_U - proposed_U) + (current_P - proposed_P)
        print('Delta: ', delta)
        alpha = np.log(np.random.random())
        print('alpha', alpha)
        if alpha > delta:
            print('Reject!')
            for j in range(len(q)):
                model.trainable_variables[j].assign(
                    q[j]
                )


        
        for steps in np.random.choice(dataset.i_train, size = 20, replace = False):
            X_d, X_l, y_d, y_l = dataset.get_index(steps)     
            x = model(X_d)
            loss = model.log_prob(x, y_l.reshape(-1,2), y_d) 
            losses[epoch, 1] += tf.math.reduce_mean( (y_d-model.sample)**2)/20.0
            losses[epoch, 2] += np.mean( (y_d < (model.m+np.sqrt(model.v) )) & 
                                          (y_d > (model.m-np.sqrt(model.v) )) )/20.0
            losses[epoch, 3] += pearsonr(model.sample.numpy().flatten(), y_d.flatten())[0] / 20.0
                
        for steps in range(n_steps_test):
            X_d, X_l, y_d, y_l = dataset.get_index(dataset.i_test[steps])     
            x = model(X_d)
            loss = model.log_prob(x, y_l.reshape(-1,2), y_d) 
            losses[epoch, 4] += tf.math.reduce_mean( (y_d-model.sample)**2)/n_steps_test
            losses[epoch, 5] += np.mean( (y_d < (model.m+np.sqrt(model.v) )) & 
                                          (y_d > (model.m-np.sqrt(model.v) )) )/n_steps_test
            losses[epoch, 6] += pearsonr(model.sample.numpy().flatten(), y_d.flatten())[0] / n_steps_test


        if epoch % print_epoch == 0:
            print('Epoch', epoch,  ' ', 'mean train loss = {:2.3f}'.format(losses[epoch,1]),
                                        'train Prob = {:.2f}'.format(losses[epoch,2]),
                                        'train correlation = {:.2f}'.format(losses[epoch,3]),
                                        '\n \t',
                                        'mean test loss = {:2.3f}'.format(losses[epoch, 4]), 
                                        'test Prob = {:.2f}'.format(losses[epoch,5]),
                                         'test correlation = {:.2f}'.format(losses[epoch,6]),
                 )
        
        if losses[epoch,6] > early_stopping_min_value:
            early_stopping_min_value = losses[epoch, 6]
            early_stopping_counter = 0
            model.save_weights('./saved_model/temp/temp_model_save')
        else:
            early_stopping_counter += 1
        if early_stopping_counter > num_early_stopping:
            print('Early Stopping at iteration {:d}'.format(epoch))
            break
    model.load_weights('./saved_model/temp/temp_model_save')
    model.save_weights('./saved_model/'+model.name+'{:.3f}'.format(early_stopping_min_value))
    return losses[:epoch, :]

def U(dataset, model):
    loss = 0.0
    for index in np.random.choice(dataset.i_train, size = 25, replace= False):
            X_d, X_l, y_d, y_l = dataset.get_index(index)
            with tf.GradientTape() as dtape:
                dtape.watch(model.trainable_variables)
                x = model(X_d.reshape(-1,3) + 1e-4*np.random.randn(X_d.shape[0], 3))
                loss += 25.0*model.log_prob(x, y_l.reshape(-1,2), y_d)/150.0
    grads = dtape.gradient(loss, model.trainable_variables)
    return loss, grads

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
    plt.colorbar(cs, cax = f3_ax4, aspect = 5, fraction = 0.01)
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
