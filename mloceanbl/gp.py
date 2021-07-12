####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
tf.keras.backend.set_floatx('float64')

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

####################################################################################
########################### Define Models ##########################################
####################################################################################

################################# Main GPR Class ###################################
class GPR(keras.Model):
    r"""
    Implements a Gaussian Process with a squared exponential kernel. The Gaussian
    Process kernel has three hyperparameters controlling the magnitude, length scale,
    and noise for the GP. If there is known input noise, that noise is included in the
    model. The hyperparameters are optimized via maximum likelihood using a set number
    of time steps (this optimization is not supposed to be perfect, just good enough). 

    Input arguments - x_l,  locations of inputs, shape (input_dim, 2)
                            columns house (lon,lat) coordinates respectively.
                    - x,    values at locations
                    - noise, Input noise, if known. Shape is same as x.

    Output arguments - m, mean of the gaussian process.
                     - V, covariance of the gaussian process.
    
    Subroutines     - sample,       produce a sample from the gaussian process
                    - log_prob,     estimate the log likelihood of the data given
                                    the gaussian process.
                    - reanalysis,   Produce the best update of mld given a model estimate
                                    and observed data.
    To do: Implement haversine distance in custom scikit learn kernel.

    """
    def __init__(self, x_l, dtype='float64', **kwargs):
        super(GPR, self).__init__(name='gaussian_process', dtype='float64', **kwargs)
        self.x_l = tf.cast(x_l, dtype='float64')
        self.x_l = tf.reshape(self.x_l, (-1, 2))
        self.input_dim = self.x_l.shape[0]
        
        self.kernel = 1.0 * Matern(length_scale=3.0, nu = 2.5, length_scale_bounds = (.25, 10))
        self.kernel_noise = 1.0 * Matern(length_scale=3.0, nu = 2.5, length_scale_bounds = (.25, 10))
        
    def fit(self, x, noise = None):
        x = tf.reshape(x, (-1,1)).numpy()
        self.gpr = GaussianProcessRegressor(kernel=self.kernel, n_restarts_optimizer = 0, random_state=0)
        self.gpr.fit(self.x_l, x)
        self.Kxx = self.gpr.kernel_(self.x_l)
        self.Kxx_chol = tf.linalg.cholesky(self.Kxx)
        
        self.gpr_noise = GaussianProcessRegressor(kernel=self.kernel_noise, n_restarts_optimizer = 0, random_state=0)
        if noise is not None:
            noise = tf.reshape(noise, (-1, 1)).numpy()
            self.gpr_noise.fit(self.x_l, noise)
            
            self.Kxx_noise = self.gpr_noise.kernel_(self.x_l)
            self.Kxx_noise_chol = tf.linalg.cholesky(self.Kxx)
    def call(self, x, y_l, noise = None, batch_size = 1, training = True):
        
        if not training:
            # Slower gaussian process regression. To do: speed up?
            self.Kyx = self.gpr.kernel_(y_l, self.x_l)
            self.Kyy = self.gpr.kernel_(y_l)
            self.m = tf.linalg.matmul(self.Kyx, tf.linalg.cholesky_solve(self.Kxx_chol, x))
            self.V = self.Kyy - tf.linalg.matmul(self.Kyx, tf.linalg.cholesky_solve(self.Kxx_chol, self.Kyx.T))

            if noise is not None:
                noise = tf.reshape(noise, (-1,))
                self.Kyx_noise = self.gpr_noise.kernel_(y_l, self.x_l)
                self.Kyy_noise = self.gpr_noise.kernel_(y_l)
                self.V += tf.linalg.matmul(
                            tf.linalg.matmul(self.Kyx_noise, 
                                             tf.linalg.cholesky_solve(self.Kxx_noise_chol, tf.linalg.diag(noise) ) ),
                            self.Kyx_noise.T)
        if training:
            # Old, fast, linear interpolation
            x_dim = self.x_l.shape[0]
            y_dim = y_l.shape[0]
            l = []
            ind = []
            for i in range(y_dim):
                d = np.sin( (self.x_l[:,1] - y_l[i,1])/180*np.pi)**2
                d += np.cos(self.x_l[:, 1]*np.pi/180.)*np.cos(y_l[i,1]*np.pi/180.)*np.sin( (self.x_l[:,0] - y_l[i,0])/180.*np.pi)**2
                d = 2*6.378*np.arcsin(np.sqrt(d))
                ind_temp = tf.where( d < 0.25 )
                d = tf.gather(d, ind_temp)
                l_temp = tf.zeros((x_dim, 1), dtype = 'float64')
                l_temp = tf.tensor_scatter_nd_update(l_temp, ind_temp, d / tf.reduce_sum(d))
                l.append(l_temp)
            self.L = tf.reshape(tf.stack(l), (y_dim, x_dim))

            self.m = tf.linalg.matmul(self.L, x)
            self.m = tf.squeeze(self.m)
            self.V = 1e-3*tf.eye(y_dim, dtype = 'float64')
            
            if noise is not None:
                self.V += tf.linalg.matmul(L, 
                            tf.linalg.matmul(tf.linalg.diag(noise),
                                        L, transpose_b = True))
            if tf.reduce_min( tf.linalg.diag(self.V)) < 1e-3:
                self.V += 1e-3*tf.eye(y_dim, dtype = 'float64')
        
        self.V_chol = tf.linalg.cholesky(self.V)
        
        return self.m, tf.linalg.diag_part(self.V)
    
    def sample(self, num_samples = 1):
        noise = tf.random.normal( shape = (int(self.m.shape[0]), int(num_samples)), dtype='float64')
        sample = tf.linalg.matmul(
                    self.V_chol,
                    noise,
                    )
        sample += tf.reshape(self.m, (-1,1))
        return tf.reshape(sample, (-1,num_samples))
    
    def log_prob(self, x, batch_size = 1):
        
        z = (self.m - tf.reshape(x, (batch_size, -1,1)))
        l = tf.reduce_mean(tf.matmul(z, 
                            tf.linalg.cholesky_solve(self.V_chol,
                                         z),
                            transpose_a = True))
        l += tf.reduce_mean(tf.math.log(tf.linalg.diag_part(self.V_chol)))
        return -l
    
    def reanalysis(self, x, y_d, noise):
        u = x
        S = tf.linalg.cholesky_solve(self.V_chol,
                                     tf.reshape(y_d, (-1,1)) - tf.linalg.matmul(self.L, 
                                                                tf.reshape(tf.gather(x, self.indexes), (-1,1))) )
        updates = tf.reshape(tf.gather(x, self.indexes), (-1,1)) + tf.linalg.matmul(tf.linalg.diag(tf.gather(noise, self.indexes)),
                                                             tf.linalg.matmul(self.L, S, transpose_a = True) )
        updates = tf.reshape(updates, (-1,))
        u = tf.tensor_scatter_nd_update(u, tf.reshape(self.indexes, (-1,1)), updates)
        return u


###################### Distance Function #########################################################
def haversine_dist(X, X2, sparse = False):
    pi = np.pi / 180
    if sparse:
        row_i = np.array([], dtype=int)
        col_i = np.array([], dtype=int)
        data = np.array([])
        for i in range(X.shape[0]):
            loc = np.argwhere(((X[i,0] - X2[:,0])**2 + (X[i,1] - X2[:,1])**2) < 5)[:,0]              
            d = np.sin(X2[loc,1]*pi - X[i,1]*pi)**2
            d += np.cos(X[i,1]*pi)*np.cos(X2[loc,1]*pi)*np.sin(X2[loc,0]*pi - X[loc,0]*pi)**2
            d = 2*6371*np.arcsin(np.sqrt(d))
            row_i = np.append(row_i, i+0*loc)
            col_i = np.append(col_i, loc)
            data = np.append(data, d)
        return (data, row_i, col_i)
    else:
        f = tf.expand_dims(X * pi, -2)  # ... x N x 1 x D
        f2 = tf.expand_dims(X2 * pi, -3)  # ... x 1 x M x D
        d = tf.sin((f - f2) / 2) ** 2
        lat1, lat2 = tf.expand_dims(X[:, 0] * pi, -1), \
                    tf.expand_dims(X2[:, 0] * pi, -2)
        cos_prod = tf.cos(lat2) * tf.cos(lat1)
        a = d[:,:,0] + cos_prod * d[:,:,1]
        c = tf.asin(tf.sqrt(a)) * 6371 * 2
        return c


