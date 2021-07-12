####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from scipy.stats import pearsonr
tf.keras.backend.set_floatx('float64')

####################################################################################
########################### Define Models ##########################################
####################################################################################
       
class CVAE(keras.Model):
    r"""
    Implements a convolutional neural network for the relationship x = f(X) + noise, 
    where X = np.vstack(X[0], X[1], ...) where x is mixed layer depth. Implements a
    variational auto-encoder. Training optimizers auto-encoder first, then optimizes
    regression network. Is suitable for EPO and SIO locations. Noise is not directly
    estimated but can be sampled.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - batch_norm, if Batch Normalization should be applied
                    - Dropout, dropout rate
                    - latent_dim, size of latent dimension in auto-encoder
                    - regressor_type, either 'CNN' or 'linear'
                    - location, whether EPO or SIO.
                            
    Output arguments - x, estimate of x = f(X) + noise
                         
    
    """
    def __init__(self, input_dim, n_features, batch_norm = True, Dropout = 0, latent_dim = 20, regressor_type = 'CNN', location = "EPO", dtype='float64', **kwargs):
        super(CVAE, self).__init__(name='variational_convolutional_auto_encoder', dtype='float64', **kwargs)
        
        # Sizes
        self.x_dim, self.y_dim = input_dim
        self.input_dim = self.x_dim*self.y_dim
        self.n_features = n_features
        self.batch_bool = batch_norm
        self.latent_dim = latent_dim
   
        # Parameters, w, b, input_noise
        # Linear weights
        if location == "EPO":
            strides = [(2,3), (2,2), (2,2)]
        else:
            strides = [(2,4), (2,3), (1,2)]
        #########################################################################################################
        # Encoder
        self.encoder = tf.keras.Sequential([])
        self.encoder.add(tf.keras.layers.InputLayer(input_shape=(self.x_dim, self.y_dim, 3)))
        for j in range(3):
            self.encoder.add(
                tf.keras.layers.Conv2D(filters = 16*2**j, kernel_size = 3, strides = strides[j], padding = 'same')
            )
            self.encoder.add(
                tf.keras.layers.LeakyReLU()
            )
            if self.batch_bool:
                self.encoder.add(
                    tf.keras.layers.BatchNormalization()
                )
            self.encoder.add(
                tf.keras.layers.Dropout(Dropout)
            )
        self.encoder.add(
          tf.keras.layers.Flatten()   
        )
        self.encoder.add(
          tf.keras.layers.Dense(2*self.latent_dim)
        )
        self.encoder.compile()
        #########################################################################################################
        # Decoder
        self.decoder = tf.keras.Sequential([])
        self.decoder.add(tf.keras.layers.InputLayer(input_shape=(self.latent_dim, )))
        self.decoder.add(
            tf.keras.layers.Dense(5*5*128)
        )
        self.decoder.add(
            tf.keras.layers.LeakyReLU(),
        )
        self.decoder.add(
            tf.keras.layers.Reshape((5, 5, 128))
        )
        for j in range(3):
            self.decoder.add(
                tf.keras.layers.Conv2DTranspose(64/(2**j), kernel_size = 3, strides = strides[2-j], padding = 'same')
            )
            self.decoder.add(
                tf.keras.layers.LeakyReLU(),
            )
        self.decoder.add(
            tf.keras.layers.Conv2DTranspose(3, kernel_size = 3, strides = 1, padding = 'same')
        )

        self.decoder.compile()
        
        #########################################################################################################
        # Regressor
        if regressor_type == 'CNN':
            self.regressor = tf.keras.Sequential([])
            self.regressor.add(tf.keras.layers.InputLayer(input_shape=(self.latent_dim, )))
            self.regressor.add(
                tf.keras.layers.Dense(5*5*128)
            )
            self.regressor.add(
                tf.keras.layers.LeakyReLU(),
            )
            self.regressor.add(
                tf.keras.layers.Reshape((5, 5, 128))
            )
            for j in range(3):
                self.regressor.add(
                    tf.keras.layers.Conv2DTranspose(64/(2**j), kernel_size = 3, strides = strides[2-j], padding = 'same')
                )
                self.encoder.add(
                    tf.keras.layers.BatchNormalization()
                )
                self.regressor.add(
                    tf.keras.layers.LeakyReLU(),
                )
            self.regressor.add(
                tf.keras.layers.Conv2DTranspose(1, kernel_size = 3, strides = 1, padding = 'same')
            )
        else:
            self.regressor = tf.keras.Sequential([])
            self.regressor.add(tf.keras.layers.InputLayer(input_shape=(self.latent_dim, )))
            self.regressor.add(
                tf.keras.layers.Dense( int(self.x_dim*self.y_dim / 4 ) )
            )
            self.regressor.add(
                    tf.keras.layers.LeakyReLU(),
            )
            self.regressor.add(
                tf.keras.layers.Dense( self.x_dim*self.y_dim )
            )
            self.regressor.add(
                tf.keras.layers.Reshape( (self.x_dim, self.y_dim) )
            )

        self.regressor.compile()
    
    @tf.function
    def sample(self, x):
        mean, logvar = tf.split(self.encoder(x, training = False), num_or_size_splits=2, axis=1)
        shape = tf.shape(mean)
        
        eps = tf.random.normal(shape=(100, shape[0], shape[1]), mean = mean, stddev = tf.exp(0.5*logvar), dtype = 'float64')
        y_ens = []
        for j in range(100):
            y_ens.append(self.regressor(eps[j]))
        
        return tf.stack(y_ens, axis = 0)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean), dtype = 'float64')
        return eps * tf.exp(logvar * .5) + mean

    def call(self, x_in, training=False):
       # Encoder
        self.x_in = x_in
        self.mean, self.logvar = tf.split(self.encoder(x_in, training = training), num_or_size_splits=2, axis=1)
        self.z = self.reparameterize(self.mean, self.logvar)
        self.x_out = self.decoder(self.z, training = training)
        x = self.regressor(self.z, training = training)
        x = tf.reshape(x, (-1, self.x_dim, self.y_dim))
        return x
    
    def log_prob(self, y_true, y_pred):
        logpx_z = -tf.reduce_mean( (self.x_in - self.x_out)**2 )
        logpz = -tf.reduce_mean(self.z**2)
        logqz_x = self.log_normal_pdf(self.z, self.mean, self.logvar)
        loss = -(logpx_z + logpz - logqz_x)
        loss += tf.reduce_mean( (y_true - y_pred)**2 )
        loss += tf.cast(tf.reduce_sum(self.losses), dtype = 'float64')
        return loss
    
    def log_normal_pdf(self, sample, mean, logvar):
        return tf.reduce_mean(
          -((sample - mean) ** 2. * tf.exp(-logvar) + logvar))
    
    def corr(self, y_true, y_pred):
        yt = y_true.numpy()
        yp = y_pred.numpy()
        batch_size = yt.shape[0]
        c = 0.0
        for j in range(batch_size):
            c += pearsonr(yt[j].flatten(), yp[j].flatten())[0]/batch_size
        return c
    
    def cal(self, x, y_true):
        sample = self.sample(x)
        m = tf.squeeze(tf.reduce_mean(sample, axis = 0))
        std = tf.squeeze(tf.math.reduce_std(sample, axis = 0))
        prob_bool = (tf.abs(y_true - m) < std)
        return tf.reduce_mean( tf.cast(prob_bool, dtype = 'float64') )
    
    def train(self, d_train, d_test, num_epochs = 10, num_epochs_split = 5, lr = 1e-3):
        opt = tf.keras.optimizers.Adam(lr)
        losses = np.zeros((6, num_epochs))
        flag = 0
        
        for epoch in range(num_epochs):
            temp_loss = 0.0
            temp_corr = 0.0
            temp_cal = 0.0
            count = 0
            for (x,y) in d_train:
                count += 1
                with tf.GradientTape() as tape:
                    y_pred = self.call(x, training = True)
                    loss = self.log_prob(y, y_pred)
                if epoch < num_epochs_split:
                    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
                else:
                    variables = self.regressor.trainable_variables
                gradients = tape.gradient(loss, variables)
                opt.apply_gradients(zip(gradients, variables))
                
                temp_loss += loss
                temp_corr += self.corr(y, y_pred)
                temp_cal += self.cal(x, y)
            
            losses[0, epoch] = temp_loss/count
            losses[1, epoch] = temp_corr/count
            losses[4, epoch] = temp_cal/count
            
            temp_loss = 0.0
            temp_corr = 0.0
            temp_cal = 0.0
            count = 0
            for (x,y) in d_test:
                count +=1
                y_pred = self.call(x, training = False)
                temp_loss += self.log_prob(y, y_pred)
                temp_corr += self.corr(y, y_pred)
                temp_cal += self.cal(x, y)
            
            losses[2, epoch] = temp_loss/count
            losses[3, epoch] = temp_corr/count
            losses[5, epoch] = temp_cal/count
            
            if (epoch % 5 == 0):
                print('Epoch: {:d} \t Training Loss: {:2.2f} \t Training Corr: {:.2f} \t Training Cal:{:.2f} \t Test Loss: {:2.2f} \t Test Corr: {:.2f} \t Test Cal: {:.2f}'.format(epoch,
                        losses[0, epoch], losses[1, epoch], losses[4, epoch], losses[2, epoch], losses[3, epoch], losses[5, epoch]))
                opt.learning_rate.assign(opt.learning_rate*0.5)
                if epoch == num_epochs_split:
                    opt.learning_rate.assign(lr)
            if (epoch > num_epochs_split + 20):
                smoothed_test_loss = [0.25*losses[3, j-1] + 0.5*losses[3, j] + 0.25*losses[3, j+1] for j in range(epoch-6, epoch-1)]
                test_loss_slope = [smoothed_test_loss[j+1] - smoothed_test_loss[j] for j in range(0, len(smoothed_test_loss)-1)]
                if (test_loss_slope[-1] < 0) & (test_loss_slope[-2] < 0):
                    print('Test Correlation is trending down')
                    flag = 1
            if flag == 1:
                losses = losses[:, :epoch]
                break
        return losses


    
