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
       
class RESNET(keras.Model):
    r"""
    Implements an deep residual neural network for the relationship x = f(X) + noise, 
    where X = np.vstack(X[0], X[1], ...). Noise can be directly estimated if
    "variational = True".
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                            
    Output arguments - x, estimate of x = f(X) + noise
                     - self.v, if "variational = True"
    """
    def __init__(self, input_dim, n_features, num_layers = 3, variational = False, Dropout = 0, location = "EPO", dtype='float64', **kwargs):
        super(RESNET, self).__init__(name='residual_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.x_dim, self.y_dim = input_dim
        self.input_dim = self.x_dim*self.y_dim
        self.n_features = n_features
        self.num_layers = num_layers
        self.variational = variational
        # Parameters, w, b, input_noise
        # Linear weights
        if location == "EPO":
            strides = (2,3)
        else:
            strides = (2,4)
        
        self.resnet_block = [ tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters = min(128, 16*2**j), kernel_size = strides, strides = 1, padding = 'same'),
            tf.keras.layers.Dropout(Dropout),
            
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters = min(128, 16*2**j), kernel_size = strides, strides = 1, padding = 'same'),

            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2D(filters = 3, kernel_size = strides, strides = 1, padding = 'same')
        ]) for j in range(num_layers)]
        
        if self.variational:
            self.cnn_final = tf.keras.layers.Conv2D(filters = 2, kernel_size = strides, strides = 1, padding = 'same')
        else:
            self.cnn_final = tf.keras.layers.Conv2D(filters = 1, kernel_size = strides, strides = 1, padding = 'same')

    def call(self, x, training=False):
        shape = tf.shape(x)
        if tf.size == 3:
            x = tf.expand_dims(x, 0)
        ## ANN Map
        for j in range(self.num_layers):
            out = self.resnet_block[j](x, training = training)
            x = x + out
        x = self.cnn_final(x)
            
        mean = tf.reshape(x[..., 0], (-1, self.x_dim, self.y_dim))
        if self.variational:
            self.v = tf.reshape(x[..., 1], (-1, self.x_dim, self.y_dim))
        return mean
    
    def model(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    def log_prob(self, y_true, y_pred):
        if self.variational:
            d = (y_true - y_pred)**2 / self.v
            d_mean = tf.reduce_mean(d)
            d_var = tf.math.reduce_variance(d)
            loss = 1e-2*(d_mean + d_var - tf.math.log(d_var))
            loss += tf.reduce_mean( (y_true - y_pred)**2 )
        else:
            loss = tf.reduce_mean( (y_true - y_pred)**2 )
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
    
    def cal(self, y_true, y_pred):
        prob_bool = ((y_true - y_pred) < tf.exp(0.5*self.v))
        return tf.reduce_mean( tf.cast(prob_bool, dtype = 'float64') )
    
    def train(self, d_train, d_test, num_epochs = 10, lr = 1e-3):
        opt = tf.keras.optimizers.Adam(lr)
        if self.variational:
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
                    variables = self.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    opt.apply_gradients(zip(gradients, variables))

                    temp_loss += loss
                    temp_corr += self.corr(y, y_pred)
                    temp_cal += self.cal(y, y_pred)

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
                    temp_cal += self.cal(y, y_pred)

                losses[2, epoch] = temp_loss/count
                losses[3, epoch] = temp_corr/count
                losses[5, epoch] = temp_cal/count

                if (epoch % 10 == 0):
                    print('Epoch: {:d} \t Training Loss: {:2.2f} \t Training Corr: {:.2f} \t Training Cal:{:.2f} \t Test Loss: {:2.2f} \t Test Corr: {:.2f} \t Test Cal: {:.2f}'.format(epoch,
                            losses[0, epoch], losses[1, epoch], losses[4, epoch], losses[2, epoch], losses[3, epoch], losses[5, epoch]))
                    opt.learning_rate.assign(opt.learning_rate*0.5)

                if (epoch > 30):
                    smoothed_test_loss = [0.25*losses[3, j-1] + 0.5*losses[3, j] + 0.25*losses[3, j+1] for j in range(epoch-6, epoch-1)]
                    test_loss_slope = [smoothed_test_loss[j+1] - smoothed_test_loss[j] for j in range(0, len(smoothed_test_loss)-1)]
                    if (test_loss_slope[-1] < 0) & (test_loss_slope[-2] < 0):
                        print('Test Correlation is trending down')
                        flag = 1
                if flag == 1:
                    losses = losses[:, :epoch]
                    break
        else:
            losses = np.zeros((4, num_epochs))
            flag = 0

            for epoch in range(num_epochs):
                temp_loss = 0.0
                temp_corr = 0.0
                count = 0
                for (x,y) in d_train:
                    count += 1
                    with tf.GradientTape() as tape:
                        y_pred = self.call(x, training = True)
                        loss = self.log_prob(y, y_pred)
                    variables = self.trainable_variables
                    gradients = tape.gradient(loss, variables)
                    opt.apply_gradients(zip(gradients, variables))

                    temp_loss += loss
                    temp_corr += self.corr(y, y_pred)

                losses[0, epoch] = temp_loss/count
                losses[1, epoch] = temp_corr/count

                temp_loss = 0.0
                temp_corr = 0.0
                count = 0
                for (x,y) in d_test:
                    count +=1
                    y_pred = self.call(x, training = False)
                    temp_loss += self.log_prob(y, y_pred)
                    temp_corr += self.corr(y, y_pred)

                losses[2, epoch] = temp_loss/count
                losses[3, epoch] = temp_corr/count

                if (epoch % 5 == 0):
                    print('Epoch: {:d} \t Training Loss: {:2.2f} \t Training Corr: {:.2f} \t Test Loss: {:2.2f} \t Test Corr: {:.2f}'.format(epoch, losses[0, epoch],
                                                                                                                                            losses[1, epoch], losses[2, epoch],
                                                                                                                                            losses[3, epoch]))
                    opt.learning_rate.assign(opt.learning_rate*0.5)

                if (epoch > 10):
                    smoothed_test_loss = [0.25*losses[3, j-1] + 0.5*losses[3, j] + 0.25*losses[3, j+1] for j in range(epoch-6, epoch-1)]
                    test_loss_slope = [smoothed_test_loss[j+1] - smoothed_test_loss[j] for j in range(0, len(smoothed_test_loss)-1)]
                    if (test_loss_slope[-1] < 0) & (test_loss_slope[-2] < 0):
                        print('Test Correlation is trending down')
                        flag = 1
                if flag == 1:
                    losses = losses[:, :epoch]
                    break
        return losses
    
