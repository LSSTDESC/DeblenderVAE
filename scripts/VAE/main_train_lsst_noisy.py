# Import necessary librairies
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras
import sys
import os
import logging
import galsim
import random
import cmath as cm
import math
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.models import Model, Sequential
from scipy.stats import norm
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten,  Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda, BatchNormalization, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import tensorflow as tf
import tensorflow_probability as tfp

from generator_vae import BatchGenerator_lsst

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions
from tools_for_VAE.callbacks import changeAlpha

######## Import data for callback (Only if VAEHistory is used)
# x = np.load('/sps/lsst/users/barcelin/data/single/galaxies_COSMOS_5_v5_test.npy')
# x_val = x[:500,1,4:]
# # Normalize the data for callback
# I= [6.48221069e+05, 4.36202878e+05, 2.27700000e+05, 4.66676013e+04,2.91513800e+02, 2.64974100e+03, 4.66828170e+03, 5.79938030e+03,5.72952590e+03, 3.50687710e+03]
# for i in range (500):
#     for j in range (6):
#         x_val[i,j] = np.tanh(np.arcsinh(x_val[i,j]/(I[4+j])))
# x_val = x_val.reshape((len(x_val),64,64,6))

######## Set some parameters
batch_size = 100
original_dim = 64*64*6
latent_dim = 10
intermediate_dim = 2000
epochs = 1000
epsilon_std = 1.0


######## Load VAE
encoder, decoder = model.vae_model(6)

######## Build the VAE
vae, vae_utils, output_encoder, Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

######## Define the loss function
alpha = K.variable(0.0001)

def vae_loss(x, x_decoded_mean):
    xent_loss = original_dim*K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
    kl = K.get_value(alpha) * Dkl
    #kl_loss = - .5 * K.get_value(alpha) * K.sum(1 + output_encoder[1] - K.square(output_encoder[0]) - K.exp(output_encoder[1]), axis=-1)
    return xent_loss + K.mean( kl_loss)

######## Compile the VAE
vae.compile('adam', loss=vae_loss, metrics=['mse'])
######## Fix the maximum learning rate in adam
K.set_value(vae.optimizer.lr, 0.0001)


#######
# Callback
alphaChanger = changeAlpha(alpha, vae, epochs)
# Callback to display evolution of training
#vae_hist = vae_functions.VAEHistory(x_val[:500], vae_utils, latent_dim, alpha, plot_bands=[2,3,5], figname='/sps/lsst/users/barcelin/callbacks/LSST/VAE/noisy/v4/test_noisy_LSST_v4')
# Keras Callbacks
#earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0000001, patience=10, verbose=0, mode='min', baseline=None)
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath='/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v7/mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath='/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v7/loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

######## Define all used callbacks
callbacks = [checkpointer_mse, checkpointer_loss]#, alphaChanger earlystop,vae_hist, 
 
######## List of data samples
list_of_samples=['/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_1_v3.npy',
                 '/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_2_v3.npy',
                 '/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_3_v3.npy',
                 '/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_4_v3.npy',
                 '/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_5_v3.npy',
                ]

######## Define the generators
training_generator = BatchGenerator_lsst(list_of_samples,total_sample_size=1800, batch_size= batch_size, size_of_lists = 40000, training_or_validation = 'training', noisy = True)#180000
validation_generator = BatchGenerator_lsst(list_of_samples,total_sample_size=200, batch_size= batch_size, size_of_lists = 40000, training_or_validation = 'validation', noisy = True)#20000


######## Train the network
hist = vae.fit_generator(generator=training_generator, epochs=epochs,
                  steps_per_epoch=18,
                  verbose=2,
                  shuffle = True,
                  validation_data=validation_generator,
                  validation_steps=2,
                  callbacks=callbacks,
                  workers = 0)

# Save the weights of last epoch 
#vae.save_weights("/pbs/throng/lsst/users/barcelin/LSST_test/v4/vae_conv_lsst_callbacks_TEST_NOISY")