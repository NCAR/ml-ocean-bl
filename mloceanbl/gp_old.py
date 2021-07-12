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
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.math.psd_kernels.internal import util
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import PositiveSemidefiniteKernel
import scipy.sparse as sp

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

    """
    def __init__(self, x_l, dtype='float64', **kwargs):
        super(GPR, self).__init__(name='gaussian_process', dtype='float64', **kwargs)
        self.x_l = tf.cast(x_l, dtype='float64')
        self.x_l = tf.reshape(self.x_l, (-1, 2))
        self.input_dim = self.x_l.shape[0]
        
        self.amplitude = tfp.util.TransformedVariable(15., 
                            tfb.Softplus(), dtype=tf.float64, name='amplitude')
        self.length_scale = tfp.util.TransformedVariable(15., 
                        tfb.Softplus(), dtype=tf.float64, name='length_scale')
        self.kernel = ExponentiatedQuadratic(self.amplitude, self.length_scale)
        self.noise_variance = tfp.util.TransformedVariable(
                    np.exp(-5), tfb.Exp(), name='observation_noise_variance')
        
        self.gp = tfd.GaussianProcess(
            kernel=self.kernel,
            index_points=self.x_l,
            observation_noise_variance=self.noise_variance)
        self.optimizer = tf.optimizers.Adam(learning_rate=.1, beta_1=.5, beta_2=.99)
    
    def train(self, x, number_of_opt_steps = 5):
        # Calculuates a gaussian process
        # m = Lx + V
        # where L = Kyx (Kxx + \sigmaI)^{-1}
        # where V = Kyy - L Kxy + L\sigmaIL^{T}
            
        for i in tf.range(number_of_opt_steps):
            neg_log_likelihood_ = self.optimize(tf.reshape(x, (-1,)), self.gp, self.optimizer)
    
    @tf.function        
    def optimize(self, x, gprm, optimizer):
        with tf.GradientTape() as tape:
            loss = -gprm.log_prob(x)
        trainable_variables = gprm.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        return loss
        
    def call(self, x, y_l, noise = None):
        x = tf.reshape(x, (-1,))
        
        ind = []
        for i in range(y_l.shape[0]):
            ind.append(tf.where( ((self.x_l[:, 0] - y_l[i, 0])**2 + (self.x_l[:,1] - y_l[i,1])**2 )<1.5 ))
        self.indexes = tf.reshape(tf.concat(ind, axis = 0), (-1,))
        inputs = tf.gather(self.x_l, self.indexes)
        x = tf.gather(x, self.indexes)    

        Kxx = self.kernel._apply(inputs, inputs)  +  \
                (self.noise_variance)*tf.eye(x.shape[0], dtype='float64')
        if noise is not None:
            noise = tf.reshape(noise, (-1,))
            noise = tf.gather(noise, self.indexes)
            noise = 1e-6+tf.math.maximum(noise, np.float64(0.0))
        Kxy = self.kernel._apply(inputs, y_l)
        Kyy = self.kernel._apply(y_l, y_l)
        self.Kyy = tf.linalg.set_diag( Kyy, tf.linalg.diag_part(Kyy) + 1e-3)
        K_chol = tf.linalg.cholesky(Kxx)
        self.L = tf.linalg.matmul(Kxy,
                            tf.linalg.cholesky_solve(K_chol, tf.eye(x.shape[0], dtype='float64')),
                            transpose_a = True)
        self.m = tf.linalg.matmul(self.L, tf.reshape(x, (-1,1)) )
                            
        self.V = self.Kyy - tf.linalg.matmul(self.L, Kxy)
        if noise is not None:
            V_temp =  tf.linalg.matmul(
                                tf.linalg.matmul(self.L, 
                                    tf.linalg.diag(noise)),
                                self.L, transpose_b = True)
            self.V += V_temp
        self.V_chol = tf.linalg.cholesky(self.V)
        
        
        return tf.reshape(self.m, (-1,)), tf.linalg.diag_part(self.V)
    
    def sample(self, num_samples = 1):
        noise = tf.random.normal( (self.m.shape[0], num_samples), dtype='float64')
        sample = self.m + tf.linalg.matmul(
                    self.V_chol,
                    noise,
                    )
        return tf.reshape(sample, (-1,1))
    
    def log_prob(self, x):
        z = (self.m - tf.reshape(x, (-1,1)))
        l = 0.5 * tf.matmul(z, 
                            tf.linalg.cholesky_solve(self.V_chol,
                                         z),
                            transpose_a = True)
        l += 0.5*x.shape[0]*tf.math.reduce_sum(tf.math.log(tf.linalg.diag_part(self.V_chol)))
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


####################### Kernel Class #########################################################
    
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
        data = haversine_dist(x1, x2)
        exponent = -0.5 * data
        
        if self.length_scale is not None:
            length_scale = tf.convert_to_tensor(self.length_scale)
            length_scale = util.pad_shape_with_ones(
              length_scale, example_ndims)
            exponent = exponent / length_scale**2
        if self.amplitude is not None:
            amplitude = tf.convert_to_tensor(self.amplitude)
            amplitude = util.pad_shape_with_ones(amplitude, example_ndims)
            exponent = exponent + 2. * tf.math.log(amplitude)

        return tf.exp(exponent) # tf.exp

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


