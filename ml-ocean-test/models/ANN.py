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
       
class ANN(keras.Model):
    r"""
    Implements an artificial neural network for the relationship x = f(S, T, H; theta) + noise, 
    where X = np.vstack(X[0], X[1], ...) where x is mixed layer depth. Is suitable for
    EPO and SIO locations. Noise is not directly estimated.
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - batch_norm, if Batch Normalization should be applied
                    - Dropout, dropout rate
                    - location, whether EPO or SIO.
                            
    Output arguments - x, estimate of x = f(X) + noise
                         
    
    """
    def __init__(self, input_dim, n_features, batch_norm = True, Dropout = 0, location = "EPO", dtype='float64', **kwargs):
        super(ANN, self).__init__(name='artificial_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.x_dim, self.y_dim = input_dim
        self.input_dim = self.x_dim*self.y_dim
        self.n_features = n_features
        self.batch_bool = batch_norm
   
        self.ann1 = tf.keras.layers.Dense( int(self.x_dim*self.y_dim / 4) )
        self.ann2 = tf.keras.layers.Dense( int(self.x_dim*self.y_dim / 8))
        self.ann3 = tf.keras.layers.Dense( int(self.x_dim*self.y_dim))
        self.flatten = tf.keras.layers.Flatten()
        self.reshape = tf.keras.layers.Reshape( (self.x_dim, self.y_dim) )
        self.batch_norm = [tf.keras.layers.BatchNormalization() for j in range(2)]
        self.dropout = tf.keras.layers.Dropout(Dropout)
        self.leaky_relu = tf.keras.layers.LeakyReLU()
        
        

    def call(self, x, training=False):
        x = self.flatten(x)
        x = self.ann1(x, training = training)
        x = self.batch_norm[0](x, training = training)
        x = self.leaky_relu(x)
        x = self.dropout(x, training = training)
        x = self.ann2(x, training = training)
        x = self.batch_norm[1](x, training = training)
        x = self.leaky_relu(x)
        x = self.ann3(x, training = training)
        x = self.reshape(x)
        return x
    
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
    
    def train(self, d_train, d_test, num_epochs = 10, lr = 1e-3):
        opt = tf.keras.optimizers.Adam(lr)
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
                    
                
