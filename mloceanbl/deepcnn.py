####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

tf.keras.backend.set_floatx('float64')
import gp


####################################################################################
########################### Define Models ##########################################
####################################################################################

class DEEPCNN(keras.Model):
    r"""
    Implements deep convolutional neural network for the model d = f(S, T, H; theta) + \epsilon
    and estimates the variance using a linear model x = \sum_{n=1}^{num_features} w[n].X[n] + b,
    with training model y = Lx + V, where L and V are obtained via GP regression.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - x_l,  locations of inputs, shape (input_dim, 2)
                            columns house (lon,lat) coordinates respectively.
                    - y_l, locations of training data
                    - y_d, training data values
                            
    Output arguments - x, estimate of x = f(X) + noise 
                     - var, variance estimate of noise
                     - m, mean estimate of m = Lx + V
                     - v, diagonal variance of gp covariance V
                        
    
    """
    def __init__(self, input_dim, n_features, x_l, dtype='float64', **kwargs):
        super(DEEPCNN, self).__init__(name='deep_convolutional_neural_network', dtype='float64', **kwargs)
        
        
        # Sizes
        self.x_dim, self.y_dim = input_dim
        self.input_dim = self.x_dim*self.y_dim
        self.n_features = n_features
        
        # Initialize grid and gaussian process        
        self.x_l = x_l
        self.gp = gp.GPR(x_l)

        if self.x_dim == 40:
            kernel_sizes = [np.asarray((2,3)), np.asarray((2,3)), np.asarray((4,6))]
        else:
            kernel_sizes = [np.asarray((1,6)), np.asarray((1,6)), np.asarray((2,12))]
        
        self.cnn_initial = tf.keras.layers.Conv2D(8, kernel_size=(2,3), strides = 1, padding = 'same')
        self.cnn1 = [self.cnn_block(16, [j*kernel_sizes[0], j*kernel_sizes[1], j*kernel_sizes[2]], dropout = 0.2) for j in range(1,3)]
        self.cnn2 = [self.cnn_block(64, [j*kernel_sizes[0], j*kernel_sizes[1], j*kernel_sizes[2]], dropout = 0.1) for j in range(1,3)]
        self.cnn3 = [self.cnn_block(128, [j*kernel_sizes[0], j*kernel_sizes[1], j*kernel_sizes[2]], dropout = 0.05) for j in range(1,3)]
        self.cnn4 = [self.cnn_block(256, [j*kernel_sizes[0], j*kernel_sizes[1], j*kernel_sizes[2]], dropout = 0.0) for j in range(1,3)]
        self.cnn_final = tf.keras.layers.Conv2D(1, kernel_size = kernel_sizes[0], strides = 1, padding = 'same')
        
        # Linear estimate of variance
        initializer = tf.keras.initializers.GlorotNormal()
        self.A2 = tf.Variable(
            initial_value=initializer(shape = (self.x_dim, self.y_dim, self.n_features)), trainable=True, validate_shape=True, caching_device=None,
            name='Weight_matrix_2', dtype='float64')

        self.b2 = tf.Variable(
            initial_value=initializer(shape = (self.x_dim, self.y_dim)), trainable=True, validate_shape=True, caching_device=None,
            name='bias_matrix_2', dtype='float64')
            
        
        self.batch = tf.keras.layers.BatchNormalization()
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        self.concat = tf.keras.layers.concatenate    
        self.reshape = tf.keras.layers.Reshape(input_dim)
        
    def cnn_block(self, filters, kernel_size, dropout = 0.1):
        model = tf.keras.Sequential([
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size[0], strides = 1, padding = 'same'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size[1], strides = 1, padding = 'same'),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size[2], strides = 1, padding = 'same'),
        ])
        return model
    
    def call(self, inputs, training=True):
        x = tf.identity(inputs)
        shape = tf.shape(x)
        if tf.size(shape) == 3:
            x = tf.expand_dims(x, 0)
        x = self.cnn_initial(x, training = training)
        x_ens1 = []
        for j in range(2):
            x_ens1.append(self.cnn1[j](x, training = training))
        x = self.concat(x_ens1)
        
        x_ens2 = []
        for j in range(2):
            x_ens2.append(self.cnn2[j](x, training = training))
        x = self.concat(x_ens1 + x_ens2)
        
        x_ens3 = []
        for j in range(2):
            x_ens3.append(self.cnn3[j](x, training = training))
        x = self.concat(x_ens1+x_ens2 + x_ens3)
        
        x_ens4 = []
        for j in range(2):
            x_ens4.append(self.cnn4[j](x, training = training))
        x = self.concat(x_ens1 + x_ens2 + x_ens3 + x_ens4)
        
        x = self.batch(x, training = training)
        x = self.leaky_relu(x)
        x = self.cnn_final(x, training = training)
        x = self.reshape(x)
        
        
        self.var = tf.math.multiply(self.A2, inputs)
        self.var = tf.reduce_sum(self.var, axis = -1) + self.b2
        self.var = 1e-8 + tf.math.softplus(self.var)
        self.var = tf.squeeze(self.var)
        
        return tf.squeeze(x)
    
    
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
        #         loss = -self.gp.log_prob( y_true )
        self.sample = self.gp.sample()
        d = (y_true - y_pred)**2 / self.v
        d_mean = tf.reduce_mean(d)
        d_var = tf.math.reduce_variance(d)
        loss = 1e-1*(d_mean + d_var - tf.math.log(d_var))
        loss += tf.reduce_mean( tf.abs(y_true - y_pred))
        loss += 1e-1*tf.reduce_mean(self.v*(self.v-1))
        return loss