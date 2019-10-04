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

import tensorflow as tf
import tensorflow_probability as tfp

from generator_deblender import BatchGenerator

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions, utils

######## Set some parameters
batch_size = 100
latent_dim = 32
epochs = 1000
bands = [0,1,2,3,4,5,6,7,8,9]

######## Import data for callback (Only if VAEHistory is used)
x = np.load('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_4_v5.npy', mmap_mode = 'c')
x_val = utils.norm(x[:500,1,bands], bands).transpose([0,2,3,1])

# Load decoder of VAE
decoder = utils.load_vae_decoder('/sps/lsst/users/barcelin/weights/LSST_EUCLID/VAE/noisy/v9/bis3/mse/',10,folder = True)
decoder.trainable = False

# Deblender model
deb_encoder, deb_decoder = model.vae_model(latent_dim, 10)

# Use the encoder of the trained VAE
deblender, deblender_utils, Dkl = vae_functions.build_vanilla_vae(deb_encoder, decoder, full_cov=False, coeff_KL = 0)

########### Comment or not depending on what's necessary
# Load weights
deblender,deblender_utils, encoder_d, Dkl = utils.load_deblender('/sps/lsst/users/barcelin/weights/LSST_EUCLID/deblender/v5/train_5/mse/', '/sps/lsst/users/barcelin/weights/LSST_EUCLID/VAE/noisy/v9/bis3/mse/', 10, folder = True)

print(deblender.summary())

# Define the loss function
alpha = K.variable(1e-2)

def deblender_loss(x, x_decoded_mean):
    xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
    kl_loss = K.get_value(alpha) * Dkl
    return xent_loss + K.mean(kl_loss)


######## Compile the deblender
deblender.compile('adam', loss=deblender_loss, metrics=['mse'])

######## Fix the maximum learning rate in adam
K.set_value(deblender.optimizer.lr, 1e-4)

#######
# Callback
path_weights = '/sps/lsst/users/barcelin/weights/LSST_EUCLID/deblender/v5/train_6/'
path_plots = '/sps/lsst/users/barcelin/callbacks/LSST_EUCLID/deblender/v5/train_6/'
#path_tb = '/sps/lsst/users/barcelin/Graph/deblender_lsst_euclid/'

#tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=path_tb+'noiseless/', histogram_freq=0, batch_size = batch_size, write_graph=True, write_images=True)
vae_hist = vae_functions.VAEHistory(x_val[:500], deblender_utils, latent_dim, alpha, plot_bands=[6], figname=path_plots+'test_')
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

######## Define all used callbacks
callbacks = [checkpointer_mse, checkpointer_loss, vae_hist]
 
######## List of data samples
list_of_samples=['/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_1_v5.npy',
                '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_2_v5.npy',
                '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_3_v5.npy',
                '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_4_v5.npy',
                '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_5_v5.npy',
                '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_6_v5.npy',
                '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_7_v5.npy']


list_of_samples_val=['/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_val_v5.npy']


######## Define the generators
training_generator = BatchGenerator(bands, list_of_samples,total_sample_size=1800, batch_size= batch_size, trainval_or_test = 'training', noisy = True)
validation_generator = BatchGenerator(bands, list_of_samples,total_sample_size=200, batch_size= batch_size, trainval_or_test = 'validation', noisy = True)

######## Train the network
hist = deblender.fit_generator(training_generator,
        epochs=epochs,
        steps_per_epoch=18,
        verbose=2,
        shuffle = True,
        validation_steps =2,
        validation_data=validation_generator, 
        callbacks=callbacks)

