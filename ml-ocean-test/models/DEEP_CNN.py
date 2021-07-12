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
       
class DEEP_CNN(keras.Model):
    r"""
    Implements a deep convolutional neural network for the relationship x = f(X) + noise, 
    where X = np.vstack(X[0], X[1], ...) where x is mixed layer depth. Is suitable for
    EPO and SIO locations. Style of CNN is deep with multiple connections between layers.
    Variance estimate can be directly outputed if "variational = True". 
    
    Input arguments - input_dim, number of rows of X
                    - n_features, number of columns of X
                    - location, whether EPO or SIO.
                    - variational, if the network should output variance estimate.
                            
    Output arguments - x, estimate of x = f(X) + noise
                     - self.v, estimate of Var[noise]. 
                         
    
    """
    def __init__(self, input_dim, n_features, variational = False, location = "EPO", dtype='float64', **kwargs):
        super(DEEP_CNN, self).__init__(name='deep_cnn_neural_network', dtype='float64', **kwargs)
        
        # Sizes
        self.x_dim, self.y_dim = input_dim
        self.input_dim = self.x_dim*self.y_dim
        self.n_features = n_features
        self.variational = variational

        if location == "EPO":
            kernel_sizes = [np.asarray((2,3)), np.asarray((2,3)), np.asarray((4,6))]
        else:
            kernel_sizes = [np.asarray((1,6)), np.asarray((1,6)), np.asarray((2,12))]
        
        self.cnn_initial = tf.keras.layers.Conv2D(8, kernel_size=(2,3), strides = 1, padding = 'same')
        self.cnn1 = [self.cnn_block(16, [j*kernel_sizes[0], j*kernel_sizes[1], j*kernel_sizes[2]], dropout = 0.2) for j in range(1,3)]
        self.cnn2 = [self.cnn_block(64, [j*kernel_sizes[0], j*kernel_sizes[1], j*kernel_sizes[2]], dropout = 0.1) for j in range(1,3)]
        self.cnn3 = [self.cnn_block(128, [j*kernel_sizes[0], j*kernel_sizes[1], j*kernel_sizes[2]], dropout = 0.05) for j in range(1,3)]
        self.cnn4 = [self.cnn_block(256, [j*kernel_sizes[0], j*kernel_sizes[1], j*kernel_sizes[2]], dropout = 0.0) for j in range(1,3)]
        self.cnn_final = tf.keras.layers.Conv2D(1, kernel_size = kernel_sizes[0], strides = 1, padding = 'same')
        if self.variational:
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
        x = self.cnn_initial(inputs, training = training)
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
        if self.variational:
            self.v = tf.math.multiply(self.A2, inputs)
            self.v = tf.reduce_sum(self.v, axis = -1) + self.b2
            self.v = 1e-8 + tf.math.softplus(self.v)
        
        return x
    
    def model(self, input_shape):
        x = tf.keras.Input(shape=input_shape)
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    def log_prob(self, y_true, y_pred):
        if self.variational:
            d = (y_true - y_pred)**2 / self.v
            d_mean = tf.reduce_mean(d)
            d_var = tf.math.reduce_variance(d)
            loss = 1e-2*(d_mean + d_var - tf.math.log(d_var))
            loss += tf.reduce_mean( tf.abs(y_true - y_pred))
        else:
            loss = tf.reduce_mean( (y_true - y_pred)**2 )
        loss += tf.cast(tf.reduce_sum(self.losses), dtype = 'float64')
        return loss
    
    def log_normal_pdf(self, sample, mean, var):
        return tf.reduce_mean(
       ((sample - mean) ** 2. /var + self.x_dim*self.y_dim*tf.math.log(var)))
    
    def corr(self, y_true, y_pred):
        yt = y_true.numpy()
        yp = y_pred.numpy()
        batch_size = yt.shape[0]
        c = 0.0
        for j in range(batch_size):
            c += pearsonr(yt[j].flatten(), yp[j].flatten())[0]/batch_size
        return c
    
    def cal(self, y_true, y_pred):
        prob_bool = (tf.abs(y_true - y_pred) < tf.sqrt(self.v))
        return tf.reduce_mean( tf.cast(prob_bool, dtype = 'float64') )
    
    def train(self, d_train, d_test,  location, datatype, num_epochs = 10, lr = 1e-3):
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

                if (epoch % 5 == 0):
                    print('Epoch: {:d} \t Training Loss: {:2.2f} \t Training Corr: {:.2f} \t Training Cal:{:.2f} \t Test Loss: {:2.2f} \t Test Corr: {:.2f} \t Test Cal: {:.2f}'.format(epoch,
                            losses[0, epoch], losses[1, epoch], losses[4, epoch], losses[2, epoch], losses[3, epoch], losses[5, epoch]))
                    self.save_weights('./models/saved_models/'+location+'/' + datatype + '/DEEPCNN/temp/deep_cnn')
                    if (epoch % 10 ==0):
                        opt.learning_rate.assign(opt.learning_rate*0.5)

                if (epoch > 20):
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
                    if (epoch % 10 ==0):
                        opt.learning_rate.assign(opt.learning_rate*0.5)

                if (epoch > 20):
                    smoothed_test_loss = [0.25*losses[3, j-1] + 0.5*losses[3, j] + 0.25*losses[3, j+1] for j in range(epoch-6, epoch-1)]
                    test_loss_slope = [smoothed_test_loss[j+1] - smoothed_test_loss[j] for j in range(0, len(smoothed_test_loss)-1)]
                    if (test_loss_slope[-1] < 0) & (test_loss_slope[-2] < 0):
                        print('Test Correlation is trending down')
                        flag = 1
                if flag == 1:
                    losses = losses[:, :epoch]
                    break
        return losses
    
