####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
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
    def __init__(self, x_l, dtype='float64', **kwargs):
        super(GPR, self).__init__(name='gaussian_process', dtype='float64', **kwargs)
        self.x_l = tf.cast(x_l, dtype='float64')
        self.input_dim = self.x_l.shape[0]
        
        
    def optimize(self, x, gprm, optimizer):
        with tf.GradientTape() as tape:
            loss = -gprm.log_prob(x)
        grads = tape.gradient(loss, gprm.trainable_variables)
        optimizer.apply_gradients(zip(grads, gprm.trainable_variables))
        return loss
        
    def call(self, x, y_l, noise = None):
        # Calculuates a gaussian process
        # m = Lx + V
        # where L = Kyx (Kxx + \sigmaI)^{-1}
        # where V = Kyy - L Kxy + L\sigmaIL^{T}
        
        self.indexes = np.array([], dtype='int')
        for i in range(y_l.shape[0]):
            ind = np.argwhere( ((self.x_l[:, 0] - y_l[i, 0])**2 + (self.x_l[:,1] - y_l[i,1])**2 )<1.5 ).flatten()
            self.indexes = np.append(self.indexes, ind)
        inputs = tf.gather(self.x_l, self.indexes)
        x = tf.gather(x, self.indexes)
        
        amplitude = tfp.util.TransformedVariable(1., 
                            tfb.Exp(), dtype=tf.float64, name='amplitude')
        length_scale = tfp.util.TransformedVariable(25., 
                        tfb.Softplus(), dtype=tf.float64, name='length_scale')
        kernel = ExponentiatedQuadratic(amplitude, length_scale)
        noise_variance = tfp.util.TransformedVariable(
                    np.exp(-5), tfb.Exp(), name='observation_noise_variance')
        
        if noise is not None
            gp = tfd.GaussianProcess(
                kernel=kernel,
                index_points=inputs,
                observation_noise_variance=noise_variance + tf.reduce_mean(noise))
        else:
            gp = tfd.GaussianProcess(
            kernel=kernel,
            index_points=inputs,
            observation_noise_variance=noise_variance)

        optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)
        for i in range(20):
            neg_log_likelihood_ = self.optimize(x, gp, optimizer)

        

        Kxx = kernel._apply(inputs, inputs)  +  \
                (noise_variance)*tf.eye(x.shape[0], dtype='float64')
        if noise is not None:
            noise = tf.gather(noise, self.indexes)
            noise = 1e-6+tf.math.maximum(noise, np.float64(0.0))
            Kxx +=  tf.linalg.diag(noise)
        Kxy = kernel._apply(inputs, y_l)
        Kyy = kernel._apply(y_l, y_l)
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
        print(tf.reshape(self.indexes, (-1,1)).shape)
        print(updates.shape)
        
        print(u.shape)
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
        length_scale = 5.0
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


