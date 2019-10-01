# Import necessary librairies
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

# from generator_deblender import BatchGenerator
sys.path.insert(0,'../VAE/')
from generator_vae import BatchGenerator

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions, utils
from tools_for_VAE.callbacks import changeAlpha

######## Set some parameters
batch_size = 128
latent_dim = 32
epochs = 100
bands = [4,5,6,7,8,9]
steps_per_epoch = 256
validation_steps = 16
load_from_vae_or_deblender = 'deblender'

images_dir = '/home/ubuntu/deblending/Deblender_VAE/images/blended/'
path_output = '/home/ubuntu/deblending/Deblender_VAE/training/deblender'
path_output_vae = '/home/ubuntu/deblending/Deblender_VAE/training/VAE'

######## Import data for callback (Only if VAEHistory is used)
# x = np.load('/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_1_v5.npy', mmap_mode = 'c')
# x_val = utils.norm(x[:500,1,bands], bands).transpose([0,2,3,1])
x_val = np.load(os.path.join(images_dir, 'validation/galaxies_blended_20190924_0_images.npy'))[:,:,bands].transpose([0,1,3,4,2])

# # Load decoder of VAE
# decoder = utils.load_vae_decoder(os.path.join(path_output_vae, 'weights'), 6, folder=True)
# decoder.trainable = False


# # Deblender model
# deb_encoder, deb_decoder = model.vae_model(latent_dim, 6)

# # Use the encoder of the trained VAE
# deblender, deblender_utils, Dkl = vae_functions.build_vanilla_vae(deb_encoder, decoder, full_cov=False, coeff_KL = 0)

########### Comment or not depending on what's necessary
# Load weights
# deblender, deblender_utils, encoder, Dkl = utils.load_vae_conv(os.path.join(path_output, 'weights'), 6, folder=True) #utils.load_deblender('/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/v5/bis2/', '/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v12/', 6, folder = True)
#K.set_value(alpha, utils.load_alpha('/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/v4/'))

if load_from_vae_or_deblender == 'vae':
    deblender, deblender_utils, encoder, decoder, Dkl = utils.load_vae_full(os.path.join(path_output_vae, 'weights'), 6, folder=True) #utils.load_deblender('/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/v5/bis2/', '/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v12/', 6, folder = True)
elif load_from_vae_or_deblender == 'deblender':
    deblender, deblender_utils, encoder, decoder, Dkl = utils.load_vae_full(os.path.join(path_output, 'weights'), 6, folder=True) #utils.load_deblender('/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/v5/bis2/', '/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v12/', 6, folder = True)
else:
    raise NotImplementedError
decoder.trainable = False

# Define the loss function
alpha = K.variable(1e-2)

def deblender_loss(x, x_decoded_mean):
    xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))#original_dim*
    kl_loss = K.get_value(alpha) * Dkl
    return xent_loss + K.mean(kl_loss)

######## Compile the VAE
deblender.compile('adam', loss=deblender_loss, metrics=['mse'])
print(deblender.summary())
######## Fix the maximum learning rate in adam
# K.set_value(deblender.optimizer.lr, 0.0001)

#######
# Callback
# path_weights = '/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/v5/bis3/'
# path_plots = '/sps/lsst/users/barcelin/callbacks/LSST/deblender/noisy/v5/bis3/'
# path_tb = '/sps/lsst/users/barcelin/Graph/deblender_lsst/'

# tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=path_tb+'noiseless/', histogram_freq=0, batch_size = batch_size, write_graph=True, write_images=True)
# vae_hist = vae_functions.VAEHistory(x_val[:500], deblender_utils, latent_dim, alpha, plot_bands=[2], figname=path_plots+'deb_lsst_v5_')
# checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
# checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

# alphaChanger = callbacks.changeAlpha(alpha, deblender, deblender_loss, path_weights)
# ######## Define all used callbacks
# callbacks = [vae_hist,checkpointer_mse]#vae_hist, , checkpointer_loss

alphaChanger = changeAlpha(alpha, deblender, deblender_loss, path_output)# path_weights)
# Callback to display evolution of training
vae_hist = vae_functions.VAEHistory(x_val[:500], deblender_utils, latent_dim, alpha, plot_bands=[1,2,3], figroot=os.path.join(path_output, 'plots/test_noisy_LSST_v4'), period=1)
# Keras Callbacks
#earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0000001, patience=10, verbose=0, mode='min', baseline=None)
# checkpointer_mse = ModelCheckpoint(filepath=os.path.join(path_output, 'weights/weights_mse_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt'), monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = ModelCheckpoint(filepath=os.path.join(path_output,'weights/weights_loss_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt'), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)

######## Define all used callbacks
callbacks = [checkpointer_loss, vae_hist, ReduceLROnPlateau(), TerminateOnNaN()]
 
######## List of data samples
# list_of_samples=['/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_1_v4.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_2_v4.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_3_v4.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_4_v4.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_5_v4.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_6_v4.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_7_v4.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_8_v4.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_9_v4.npy']

# list_of_samples=['/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_1_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_2_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_3_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_4_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_5_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_6_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_7_v5.npy']


# list_of_samples_val=['/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_val_v5.npy']

list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
# print("main_train")
# print(list_of_samples)

# list_of_samples_val = ['/sps/lsst/users/barcelin/data/single/PSF_lsst_O.65/independant/galaxies_COSMOS_val_v5_test.npy']
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]

######## Define the generators
# training_generator = BatchGenerator(bands,list_of_samples,total_sample_size=180000, batch_size= batch_size, trainval_or_test = 'training', noisy = True)#190000
# validation_generator = BatchGenerator(bands,list_of_samples_val,total_sample_size=20000, batch_size= batch_size, trainval_or_test = 'validation', noisy = True)#10000
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
# hist = deblender.fit_generator(training_generator,
#         epochs=epochs,
#         steps_per_epoch=18,#1900,
#         verbose=2,
#         shuffle = True,
#         validation_steps =2,#100,
#         validation_data=validation_generator,
#         callbacks=callbacks,
#         workers = 0)
hist = deblender.fit_generator(generator=training_generator, epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  verbose=1,
                  shuffle=True,
                  validation_data=validation_generator,
                  validation_steps=validation_steps,
                  callbacks=callbacks,
                  workers=0)