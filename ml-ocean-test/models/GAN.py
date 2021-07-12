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
       
class GAN(keras.Model):
    r"""
    Implements an GAN model for x = f(X) + noise, 
    where X = np.vstack(X[0], X[1], ...). Noise is not estimated directly but can 
    be sampled through the GAN.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                            
    Output arguments - x, estimate of x = f(X) + noise
    
    
    """
    def __init__(self, input_dim, n_features, batch_norm = True, Dropout = 0.1, latent_dim = 20, location = "EPO", dtype='float64', **kwargs):
        super(GAN, self).__init__(name='generative_network', dtype='float64', **kwargs)
        
        # Sizes
        self.x_dim, self.y_dim = input_dim
        self.input_dim = self.x_dim*self.y_dim
        self.n_features = n_features
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
                    tf.keras.layers.BatchNormalization()
            )
            self.encoder.add(
                tf.keras.layers.LeakyReLU()
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
        # Generator
        self.generator = tf.keras.Sequential([])
        self.generator.add(tf.keras.layers.InputLayer(input_shape=(self.latent_dim, )))
        self.generator.add(
            tf.keras.layers.Dense(5*5*128)
        )
        self.generator.add(
            tf.keras.layers.LeakyReLU(),
        )
        self.generator.add(
            tf.keras.layers.Reshape((5, 5, 128))
        )
        for j in range(3):
            self.generator.add(
                tf.keras.layers.Conv2DTranspose(64/(2**j), kernel_size = 3, strides = strides[2-j], padding = 'same')
            )
            self.encoder.add(
                    tf.keras.layers.BatchNormalization()
            )
            self.generator.add(
                tf.keras.layers.LeakyReLU(),
            )
        self.generator.add(
            tf.keras.layers.Conv2DTranspose(1, kernel_size = 3, strides = 1, padding = 'same')
        )
        self.generator.add(tf.keras.layers.Reshape((self.x_dim, self.y_dim)))
        self.generator.compile()
        
        #########################################################################################################
        # Discriminator
        self.discriminator = tf.keras.Sequential([])
        self.discriminator.add(tf.keras.layers.InputLayer(input_shape=(self.x_dim, self.y_dim)))
        self.discriminator.add(tf.keras.layers.Reshape((self.x_dim, self.y_dim, 1)))
        for j in range(3):
            self.discriminator.add(
                tf.keras.layers.Conv2D(filters = 16*2**j, kernel_size = 3, strides = strides[j], padding = 'same')
            )
            self.encoder.add(
                    tf.keras.layers.BatchNormalization()
            )
            self.discriminator.add(
                tf.keras.layers.LeakyReLU()
            )
            self.discriminator.add(
                tf.keras.layers.Dropout(Dropout)
            )
        self.discriminator.add(
          tf.keras.layers.Flatten()   
        )
        self.discriminator.add(
          tf.keras.layers.Dense(1)
        )
        self.discriminator.compile()
    
    def call(self, x, training=False):
       # Encoder
        mean, logvar = tf.split(self.encoder(x, training = training), num_or_size_splits=2, axis=1)
        z = self.reparameterize(mean, logvar)
        x = self.generator(z, training = training)
        return x
    
    @tf.function
    def sample(self, x):
        mean, logvar = tf.split(self.encoder(x, training = False), num_or_size_splits=2, axis=1)
        shape = tf.shape(mean)
        
        eps = tf.random.normal(shape=(100, shape[0], shape[1]), mean = mean, stddev = tf.exp(0.5*logvar), dtype = 'float64')
        y_ens = []
        for j in range(100):
            y_ens.append(self.generator(eps[j]))
        
        return tf.stack(y_ens, axis = 0)

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean), dtype = 'float64')
        return eps * tf.exp(logvar * .5) + mean

    def discriminator_loss(self, real_output, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return 0.5*total_loss
    
    def generator_loss(self, fake_output):
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    @tf.function
    def train_step(self, x, y, encoder_optimizer, generator_optimizer, discriminator_optimizer):
        with tf.GradientTape() as enc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            mean, logvar = tf.split(self.encoder(x, training = True), num_or_size_splits=2, axis=1)
            z = self.reparameterize(mean, logvar)
            
            y_pred = self.generator(z, training = True)
            
            real_output = self.discriminator(y, training=True)
            fake_output = self.discriminator(y_pred, training=True)

            gen_loss = self.generator_loss(fake_output)
            gen_loss += tf.reduce_mean( (y_pred - y)**2 )
            gen_loss += tf.reduce_mean(z**2 - (z - mean)**2*tf.exp(-logvar)  + logvar)
            gen_loss += tf.reduce_mean(tf.exp(logvar) - logvar)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        gradients_of_encoder = enc_tape.gradient(gen_loss, self.encoder.trainable_variables)
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        encoder_optimizer.apply_gradients(zip(gradients_of_encoder, self.encoder.trainable_variables))
        generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
        return gen_loss + disc_loss
    
    def log_prob(self, y, y_pred):
        real_output = self.discriminator(y)
        fake_output = self.discriminator(y_pred)
        gen_loss = self.generator_loss(fake_output)
        disc_loss = self.discriminator_loss(real_output, fake_output)
        return gen_loss, disc_loss
    
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
        m = tf.reduce_mean(sample, axis = 1)
        std = tf.math.reduce_std(sample, axis = 1)
        prob_bool = (tf.abs(y_true[:, None, ...] - m) < std)
        return tf.reduce_mean( tf.cast(prob_bool, dtype = 'float64') )
    
    def train(self, d_train, d_test, num_epochs = 10, lr = 1e-3):
        opt = tf.keras.optimizers.Adam(lr)
        losses = np.zeros((7, num_epochs))
        flag = 0
        encoder_optimizer = tf.keras.optimizers.Adam(0.5*lr)
        generator_optimizer = tf.keras.optimizers.Adam(lr)
        discriminator_optimizer = tf.keras.optimizers.Adam(lr)
        for epoch in range(num_epochs):
            temp_loss = 0.0
            temp_corr = 0.0
            temp_cal = 0.0
            count = 0
            for (x,y) in d_train:
                count += 1
                loss = self.train_step(x, y, encoder_optimizer, generator_optimizer, discriminator_optimizer)
                y_pred = self.call(x, training = True)
                temp_loss += loss
                temp_corr += self.corr(y, y_pred)
                temp_cal += self.cal(x, y)
            
            losses[0, epoch] = temp_loss/count
            losses[1, epoch] = temp_corr/count
            losses[4, epoch] = temp_cal/count
            
            temp_loss_gen = 0.0
            temp_loss_disc = 0.0
            temp_corr = 0.0
            temp_cal = 0.0
            count = 0
            for (x,y) in d_test:
                count +=1
                y_pred = self.call(x, training = False)
                temp = self.log_prob(y, y_pred)
                temp_loss_gen += temp[0]
                temp_loss_disc += temp[1]
                temp_corr += self.corr(y, y_pred)
                temp_cal += self.cal(x, y)
            
            losses[2, epoch] = temp_loss_gen/count
            losses[6, epoch] = temp_loss_disc/count
            losses[3, epoch] = temp_corr/count
            losses[5, epoch] = temp_cal/count
            
            if (epoch % 5 == 0):
                print('Epoch: {:d} \t Training Loss: {:2.2f} \t Training Corr: {:.2f} \t Training Cal:{:.2f} \t Test Loss: {:2.2f} \t Test Corr: {:.2f} \t Test Cal: {:.2f}'.format(epoch,
                        losses[0, epoch], losses[1, epoch], losses[4, epoch], losses[2, epoch], losses[3, epoch], losses[5, epoch]))
                if (epoch + 1) % 50 == 0:
                    opt.learning_rate.assign(opt.learning_rate*0.7)
#             if (epoch > 10):
#                 smoothed_test_loss = [0.25*losses[3, j-1] + 0.5*losses[3, j] + 0.25*losses[3, j+1] for j in range(epoch-6, epoch-1)]
#                 test_loss_slope = [smoothed_test_loss[j+1] - smoothed_test_loss[j] for j in range(0, len(smoothed_test_loss)-1)]
#                 if (test_loss_slope[-1] < 0) & (test_loss_slope[-2] < 0):
#                     print('Test Correlation is trending down')
#                     flag = 1
            if flag == 1:
                losses = losses[:, :epoch]
                break
        return losses


    
