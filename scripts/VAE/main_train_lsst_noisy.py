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

from generator_vae import BatchGenerator

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions, utils
from tools_for_VAE.callbacks import changeAlpha

######## Set some parameters
batch_size = 128
latent_dim = 32
epochs = 10000
bands = [4,5,6,7,8,9]

images_dir = '/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/'
path_output = '/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v12/bis4/'

######## Import data for callback (Only if VAEHistory is used)
x_val = np.load(os.path.join(images_dir, 'galaxies_COSMOS_5_v5_test.npy'))[:,:,bands].transpose([0,1,3,4,2])

######## Load VAE
encoder, decoder = model.vae_model(latent_dim, len(bands))

######## Build the VAE
vae, vae_utils, Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

######## Comment or not depending on what's necessary
# Load weights
vae, vae_utils, encoder, Dkl = utils.load_vae_conv(os.path.join(path_output, 'weights'), 6, folder=True) 
#K.set_value(alpha, utils.load_alpha('/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v10/'))

print(vae.summary())

######## Define the loss function
alpha = K.variable(1e-2)

def vae_loss(x, x_decoded_mean):
    xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
    kl_loss = K.get_value(alpha) * Dkl
    return xent_loss + K.mean(kl_loss)


######## Compile the VAE
vae.compile('adam', loss=vae_loss, metrics=['mse'])

######## Fix the maximum learning rate in adam
# K.set_value(vae.optimizer.lr, 0.0001)


#######
# Callback
path_weights = '/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v12/bis5/'
path_plots = '/sps/lsst/users/barcelin/callbacks/LSST/VAE/noisy/v12/bis5/'
path_tb = '/sps/lsst/users/barcelin/Graph/vae_lsst_r_band/noisy/'

alphaChanger = changeAlpha(alpha, vae, vae_loss, path_output)# path_weights)
# Callback to display evolution of training
vae_hist = vae_functions.VAEHistory(x_val[:500], vae_utils, latent_dim, alpha, plot_bands=[1,2,3], figroot=os.path.join(path_output, 'plots/test_noisy_LSST_v4'), period=5)
# Keras Callbacks
#earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0000001, patience=10, verbose=0, mode='min', baseline=None)
checkpointer_mse = ModelCheckpoint(filepath=os.path.join(path_output, 'weights/weights_mse_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt'), monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
#checkpointer_loss = ModelCheckpoint(filepath=os.path.join(path_output,'weights/weights_loss_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

######## Define all used callbacks
callbacks = [checkpointer_mse, vae_hist, ReduceLROnPlateau(), TerminateOnNaN()]# checkpointer_mse earlystop, checkpointer_loss,vae_hist,, alphaChanger

######## Create generators
#list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
#list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]

list_of_samples=['/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_1_v5_test.npy',
                  '/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_2_v5_test.npy',
                  '/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_3_v5_test.npy',
                  '/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_4_v5_test.npy',
                  '/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_5_v5_test.npy',
                  '/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_6_v5_test.npy',
                  '/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_7_v5_test.npy'#,
                ]

list_of_samples_val = ['/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_val_v5_test.npy']


training_generator = BatchGenerator(bands, list_of_samples, total_sample_size=None,
                                    batch_size=batch_size, size_of_lists=None,
                                    scale_radius=None, SNR=None,
                                    trainval_or_test='training',
                                    noisy=True, do_norm=False)#180000

validation_generator = BatchGenerator(bands, list_of_samples_val, total_sample_size=None,
                                    batch_size=batch_size, size_of_lists=None,
                                    scale_radius=None, SNR=None,
                                    trainval_or_test='validation',
                                    noisy=True, do_norm=False)#180000

######## Train the network
hist = vae.fit_generator(generator=training_generator, epochs=epochs,
                  steps_per_epoch=256,
                  verbose=1,
                  shuffle=True,
                  validation_data=validation_generator,
                  validation_steps=16,
                  callbacks=callbacks,
                  workers=0)
