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
       
class CNND(keras.Model):
    r"""
    Implements a convolutional neural network for the relationship x = f(X) + noise, 
    where X = np.vstack(X[0], X[1], ...) where x is mixed layer depth. Is suitable for
    EPO and SIO locations. Noise is not directly estimated but can be sampled through Dropout.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - batch_norm, if Batch Normalization should be applied
                    - num_layers, number of conv layers
                    - Dropout, dropout rate (also applied during testing)
                    - num_dense_layers, number of dense layers to incldue at end of conv layers
                    - location, whether EPO or SIO.
                            
    Output arguments - x, estimate of x = f(X) + noise
                         
    
    """
    def __init__(self, input_dim, n_features, batch_norm = True, num_layers = 3, Dropout = 0.2, num_dense_layers = 3, location = "EPO", dtype='float64', **kwargs):
        super(CNND, self).__init__(name='dropout_convolutional_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.x_dim, self.y_dim = input_dim
        self.input_dim = self.x_dim*self.y_dim
        self.n_features = n_features
        self.batch_bool = batch_norm
        self.num_layers = num_layers
        self.num_dense_layers = num_dense_layers
   
        
        if num_dense_layers == 0:
            if location == "EPO":
                strides = (2,3)
            else:
                strides = (2,4)
            self.cnn = [tf.keras.layers.Conv2D(filters = min(16*2**j, 128), kernel_size = strides, strides = 1, padding = 'same') for j in range(num_layers)]
            self.cnn_final = tf.keras.layers.Conv2D(filters = 1, kernel_size = strides, strides = 1, padding = 'same')
        else:
            if location == "EPO":
                strides = [(2,3), (2,2), (2,2)]
            else:
                strides = [(2,4), (2,3), (1,2)]
            self.cnn = [tf.keras.layers.Conv2D(filters = min(16*2**j, 128), kernel_size = 3, strides = strides[j], padding = 'same') for j in range(num_layers)]
            self.dense_layers = [tf.keras.layers.Dense( int(self.x_dim*self.y_dim / 2**(num_dense_layers-j-1) ) ) for j in range(num_dense_layers)]
            self.flatten = tf.keras.layers.Flatten()
        self.batch_norm = [tf.keras.layers.BatchNormalization() for j in range(num_layers)]
        self.dropout = [tf.keras.layers.Dropout(Dropout) for j in range(num_layers)]
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        
        

    def call(self, x, training=False):

        for j in range(self.num_layers):
            x = self.cnn[j](x, training = training)
            x = self.leaky_relu(x, training = training)
            if self.batch_bool:
                x = self.batch_norm[j](x, training = training)
            # Monte carlo dropout
            x = self.dropout[j](x, training = True)
        if self.num_dense_layers == 0:
            x = self.cnn_final(x)      
            mean = tf.reshape(x[..., 0], (-1, self.x_dim, self.y_dim))
        else:
            x = self.flatten(x)
            for j in range(self.num_dense_layers-1):
                x = self.dense_layers[j](x)
                x = self.leaky_relu(x)
            x = self.dense_layers[-1](x)
            mean = tf.reshape(x, (-1, self.x_dim, self.y_dim))
        return mean
    
    @tf.function
    def sample(self, x):
        shape = tf.shape(x)
        eps = tf.random.normal(shape=(100, shape[0], shape[1], shape[2], shape[3]), mean = x, stddev = 1e-2, dtype = 'float64')
        y_ens = []
        for j in range(100):
            y_ens.append(self.call(eps[j]))
        
        return tf.stack(y_ens, axis = 0)

    
    def model(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    def log_prob(self, y_true, y_pred):
        loss = tf.reduce_mean( (y_true - y_pred)**2 )
        loss += tf.cast(tf.reduce_sum(self.losses), dtype = 'float64')
        return loss
    
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
        m = tf.reduce_mean(sample, axis = 0)
        std = tf.math.reduce_std(sample, axis = 0)
        prob_bool = ((y_true - m) < std)
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
                variables = self.trainable_variables
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
            if (epoch > num_epochs_split + 10):
                smoothed_test_loss = [0.25*losses[3, j-1] + 0.5*losses[3, j] + 0.25*losses[3, j+1] for j in range(epoch-6, epoch-1)]
                test_loss_slope = [smoothed_test_loss[j+1] - smoothed_test_loss[j] for j in range(0, len(smoothed_test_loss)-1)]
                if (test_loss_slope[-1] < 0) & (test_loss_slope[-2] < 0):
                    print('Test Correlation is trending down')
                    flag = 1
            if flag == 1:
                losses = losses[:, :epoch]
                break
        return losses
                    
                
