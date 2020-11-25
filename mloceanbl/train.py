####################################################################################
####################################################################################
########################### File Description #######################################
####################################################################################
####################################################################################

# The collection of training functions. They are organized as follows:
#         1. train (dataset, models, ...)
#             is the main function that takes a dataset and a model (with keyword
#             parameters) and trains via minibatches using an MCMC-style acceptance.
#             Importantly, it also calculates and prints losses after each epoch.

#         2. train_minibatch(dataset, model, mini_batch_size, optimizer)
#             a helper function that handles the minibatches.

#         3. batch_train_mcmc( ... )
#             a helper function that calculates the loss for the first batch and
#             accepts an update only if the new loss is less than the first loss or
#             with probability exp(first loss - new loss)



####################################################################################
########################### Import libraries #######################################
####################################################################################
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import pearsonr
tf.keras.backend.set_floatx('float64')
tfd = tfp.distributions
tfb = tfp.bijectors

####################################################################################
########################### Training Proceedure ####################################
####################################################################################
def train(dataset, model, epochs = 500, print_epoch = 100, lr = 0.001, num_early_stopping = 500, num_restarts = 3, mini_batch_size=10):
    losses = np.zeros((epochs,6))
    
    n_steps_train = dataset.i_train.size
    n_steps_test = dataset.i_test.size

    optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
    restart_counter = 0
    
    # Implement early_stopping
    early_stopping_counter = 0
    early_stopping_min_value = 0
    stop = False
    
    # Iterate over epochs.
    for epoch in range(epochs):
        
        train_minibatch(dataset, model, mini_batch_size, optimizer)
        
        for steps in np.random.choice(dataset.i_train, size = 21, replace = False):
            X_d, X_l, y_d, y_l = dataset.get_index(steps)     
            x = model(X_d)
            loss = model.log_prob(x, y_l.reshape(-1,2), y_d) 
            losses[epoch, 0] += tf.math.reduce_mean( (y_d-model.sample)**2)/21.0
            losses[epoch, 1] += np.mean( (y_d < (model.m+np.sqrt(model.v) )) & 
                                          (y_d > (model.m-np.sqrt(model.v) )) )/21.0
            losses[epoch, 2] += pearsonr(model.sample.numpy().flatten(), y_d.flatten())[0] / 21.0
                
        for steps in range(n_steps_test):
            X_d, X_l, y_d, y_l = dataset.get_index(dataset.i_test[steps])     
            x = model(X_d)
            loss = model.log_prob(x, y_l.reshape(-1,2), y_d) 
            losses[epoch, 3] += tf.math.reduce_mean( (y_d-model.sample)**2)/n_steps_test
            losses[epoch, 4] += np.mean( (y_d < (model.m+np.sqrt(model.v) )) & 
                                          (y_d > (model.m-np.sqrt(model.v) )) )/n_steps_test
            losses[epoch, 5] += pearsonr(model.sample.numpy().flatten(), y_d.flatten())[0] / n_steps_test


        if epoch % print_epoch == 0:
            print('Epoch', epoch,  ' ', 'mean train loss = {:2.3f}'.format(losses[epoch,0]),
                                        'train Prob = {:.2f}'.format(losses[epoch,1]),
                                        'train correlation = {:.2f}'.format(losses[epoch,2]),
                                        '\n \t',
                                        'mean test loss = {:2.3f}'.format(losses[epoch, 3]), 
                                        'test Prob = {:.2f}'.format(losses[epoch,4]),
                                         'test correlation = {:.2f}'.format(losses[epoch,5]),
                 )
        
        if losses[epoch,5] > early_stopping_min_value:
            early_stopping_min_value = losses[epoch, 5]
            early_stopping_counter = 0
            model.save_weights('./saved_model/temp/temp_model_save')
        else:
            early_stopping_counter += 1
        if (early_stopping_counter > num_early_stopping):
            if (restart_counter < num_restarts):
                print('Early Stopping at iteration {:d}'.format(epoch))
                print('Restarting with smaller learning_rate')
                model.load_weights('./saved_model/temp/temp_model_save')
                lr = lr/4.0
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                restart_counter += 1
                early_stopping_counter = 0
            else:
                print('Final Early Stopping at iteration {:d}'.format(epoch))
                stop = True
        if stop:
            break
    model.load_weights('./saved_model/temp/temp_model_save')
    model.save_weights('./saved_model/'+model.name+'{:.3f}'.format(early_stopping_min_value))
    return losses[:epoch, :]

def train_minibatch(dataset, model, mini_batch_size, optimizer):
    n_steps_train = dataset.i_train.size
    batch_index = np.random.permutation(n_steps_train)
    num_minibatch = np.ceil(n_steps_train / mini_batch_size).astype(int)

    X_d_test_batch = np.zeros((mini_batch_size, model.input_dim, model.n_features))
    y_d_test_batch = []
    y_l_test_batch1 = []
    y_l_test_batch2 = []
    for j in range(1, 1+mini_batch_size):
        X_d, X_l, y_d, y_l = dataset.get_index(dataset.i_train[batch_index[-j]])
        X_d_test_batch[j-1] = X_d
        y_d_test_batch.append(y_d.reshape(-1,))
        y_l_test_batch1.append(y_l[:,0])
        y_l_test_batch2.append(y_l[:,1])
    X_d_test_batch = tf.cast(X_d_test_batch, dtype='float64')
    y_d_test_batch = tf.ragged.constant(y_d_test_batch, ragged_rank=1)
    y_l_test_batch = tf.stack((tf.ragged.constant(y_l_test_batch1, ragged_rank=1),
                        tf.ragged.constant(y_l_test_batch2, ragged_rank=1)), axis=-1)
    i = 0
    for batch in range(num_minibatch - 1):
        X_d_batch = np.zeros((mini_batch_size, model.input_dim, model.n_features))
        y_d_batch = []
        y_l_batch1 = []
        y_l_batch2 = []
        for j in range(mini_batch_size):
            X_d, X_l, y_d, y_l = dataset.get_index(dataset.i_train[batch_index[i]])
            X_d_batch[j] = X_d
            y_d_batch.append(y_d.reshape(-1,))
            y_l_batch1.append(y_l[:,0])
            y_l_batch2.append(y_l[:,1])
            i+=1
        X_d_batch = tf.cast(X_d_batch, dtype='float64')
        y_d_batch = tf.ragged.constant(y_d_batch, ragged_rank=1)
        y_l_batch = tf.stack((tf.ragged.constant(y_l_batch1, ragged_rank=1),
                            tf.ragged.constant(y_l_batch2, ragged_rank=1)), axis=-1)
        batch_train_mcmc(optimizer, model, X_d_batch, y_d_batch, y_l_batch, X_d_test_batch, y_d_test_batch, y_l_test_batch, mini_batch_size, n_steps_train)

def batch_train_mcmc(optimizer, model, X_d_batch, y_d_batch, y_l_batch, X_d_test_batch, y_d_test_batch, y_l_test_batch, mini_batch_size, n_steps_train):
    old_weights = model.save_weights('./saved_model/training_temp/temp')
    test_loss = 0.0
    for j in range(mini_batch_size):
        X_d = X_d_test_batch[j]
        y_d = y_d_test_batch[j]
        y_l = y_l_test_batch[j]
        x = model(X_d + tf.random.normal( (X_d.shape[0], 3), stddev=1e-4, dtype=tf.float64) )
        test_loss += mini_batch_size*model.log_prob(x, y_l, y_d)/n_steps_train
    
    loss = 0.0
    for j in range(mini_batch_size):
        with tf.GradientTape() as dtape:
            X_d = X_d_batch[j]
            y_d = y_d_batch[j]
            y_l = y_l_batch[j]
            dtape.watch(model.trainable_variables)
            x = model(X_d + tf.random.normal( (X_d.shape[0], 3), stddev=1e-4, dtype=tf.float64) )
            loss += mini_batch_size*model.log_prob(x, y_l, y_d)/n_steps_train
    # Apply gradients to model parameters
    grads = dtape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    new_test_loss = 0.0
    for j in range(mini_batch_size):
        X_d = X_d_test_batch[j]
        y_d = y_d_test_batch[j]
        y_l = y_l_test_batch[j]
        x = model(X_d + tf.random.normal( (X_d.shape[0], 3), stddev=1e-4, dtype=tf.float64) )
        new_test_loss += mini_batch_size*model.log_prob(x, y_l, y_d)/n_steps_train

    alpha = tf.cast( (test_loss - new_test_loss)/np.sqrt(np.abs(test_loss)) , dtype='float64')
    if np.float64(np.random.rand()) > tf.exp(alpha):
        print('Reject!')
        model.load_weights('./saved_model/training_temp/temp')

    
    


            
    

