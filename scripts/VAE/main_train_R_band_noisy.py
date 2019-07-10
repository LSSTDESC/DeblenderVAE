# Import necessary librairies
import numpy as np
import matplotlib.pyplot as plt
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

from generator_vae import BatchGenerator_lsst_r_band, BatchGenerator

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import vae_functions, model, utils
from tools_for_VAE.callbacks import changeAlpha


######## Set some parameters
batch_size = 100
latent_dim = 32
epochs = 1000
bands = [6]

######## Import data for callback (Only if VAEHistory is used)
x = np.load('/sps/lsst/users/barcelin/data/single/galaxies_COSMOS_5_v5_test.npy', mmap_mode = 'c')
x_val = utils.norm(np.expand_dims(x[:500,1,6], axis = 1), bands).transpose([0,2,3,1])

######## Load encoder, decoder
encoder, decoder = model.vae_model(latent_dim, 1)

######## Build the VAE
vae, vae_utils,  Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)
print(vae.summary())

######## Define the loss function
alpha = K.variable(1e-4)

def vae_loss(x, x_decoded_mean):
     xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
     kl_loss =  K.get_value(alpha) * Dkl
     return xent_loss + K.mean(kl_loss)

######## Compile the VAE
vae.compile('adam', loss=vae_loss, metrics=['mse'])

######## Fix the maximum learning rate in adam
K.set_value(vae.optimizer.lr, 0.0001)

#######
# Callback
path_weights = '/sps/lsst/users/barcelin/weights/R_band/VAE/noisy/test_KL/test_3/'
path_plots = '/sps/lsst/users/barcelin/callbacks/R_band/VAE/noisy/test_kl/test_3/'
path_tb = '/sps/lsst/users/barcelin/Graph/vae_lsst_r_band/noisy/'
#alphaChanger = changeAlpha(alpha, vae, epochs)
# Callback to display evolution of training
vae_hist = vae_functions.VAEHistory(x_val, vae_utils, latent_dim, alpha, plot_bands=0, figname=path_plots+'test_')#noisy_
# Keras Callbacks
#earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0000001, patience=10, verbose=0, mode='min', baseline=None)
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'weights.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath='/sps/lsst/users/barcelin/weights/R_band/VAE/noisy/test_kl/loss/weights.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
#tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=path_tb+'v6', histogram_freq=0, batch_size = batch_size, write_graph=True, write_images=True)

######## Define all used callbacks
callbacks = [vae_hist, checkpointer_mse]#,checkpointer_loss, tbCallBack]#, alphaChanger earlystop,vae_hist, checkpointer,  
 
######## List of data samples
list_of_samples=['/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_1_v3.npy',
                 '/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_2_v3.npy',
                 '/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_3_v3.npy',
                 '/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_4_v3.npy',
                 '/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_5_v3.npy',
                ]
list_of_samples_val = ['/sps/lsst/users/barcelin/data/single/v7/galaxies_COSMOS_5_v3_val.npy']

######## Define the generators
training_generator = BatchGenerator(bands, list_of_samples,total_sample_size=180000, batch_size= batch_size, size_of_lists = 40000, training_or_validation = 'training', noisy = True)#180000
validation_generator = BatchGenerator(bands, list_of_samples_val,total_sample_size=20000, batch_size= batch_size, size_of_lists = 40000, training_or_validation = 'validation', noisy = True)#20000

######## Train the network
hist = vae.fit_generator(generator=training_generator, epochs=epochs,
                  steps_per_epoch=18,#1800
                  verbose=2,
                  shuffle = True,
                  validation_data=validation_generator,
                  validation_steps=2,#200
                  callbacks=callbacks,
                  workers = 0)
