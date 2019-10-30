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

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions, utils
from tools_for_VAE.callbacks import changeAlpha
from tools_for_VAE.generator import BatchGenerator

######## Set some parameters
batch_size = 128
latent_dim = 32
epochs = 10000
bands = [4,5,6,7,8,9]

images_dir = '/sps/lsst/users/barcelin/data/single_galaxies/'
path_output = '/sps/lsst/users/barcelin/test_VAE/'

######## Import data for callback (Only if VAEHistory is used)
x_val = np.load(os.path.join(images_dir, 'validation/galaxies_isolated_20191022_0_images.npy'))[:,:,bands].transpose([0,1,3,4,2])

######## Load VAE
encoder, decoder = model.vae_model_2(latent_dim, len(bands), last_activation='softplus')
encoder.summary()
decoder.summary()

######## Build the VAE
vae, vae_utils, Dkl = vae_functions.build_vanilla_vae(encoder, decoder, full_cov=False, coeff_KL = 0)

######## Comment or not depending on what's necessary
# Load weights
#vae, vae_utils, encoder, Dkl = utils.load_vae_conv(os.path.join(path_output, 'weights_loss'), 6, folder=True, last_activation='softplus')

# #K.set_value(alpha, utils.load_alpha('/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v10/'))

print(vae.summary())

######## Define the loss function
alpha = K.variable(0.01)

sky_level_pixel = [521.2193897195763,
 561.4534053337984,
 1811.052086665226,
 352.64353501116443,
 1578.414111061399,
 15931.997840778231,
 35504.82050297898,
 50850.47090068849,
 79123.30214621601,
 90846.321995405]

def vae_loss(x, x_decoded_mean):
    #xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
    logit_x = K.log(x/(1-x)) #K.clip(x,0.,1e4)
    mu = x_decoded_mean #K.clip(x_decoded_mean,0.,1e4)
    sigma2 = mu
    xent_loss = K.clip(K.mean(K.sum(K.log(sigma2)/2. + K.square(logit_x-mu)/(2*sigma2) + K.log(1/(x*(1-x))), axis=[1,2,3])), 0., 5e5)
    kl_loss = K.get_value(alpha) * K.clip(Dkl, 0., 5e5)
    return xent_loss + K.mean(kl_loss)

######## Compile the VAE
#from tensorflow.keras.optimizers import Adam​

vae.compile(optimizer='adam', loss=vae_loss, metrics=['mse'])

######## Fix the maximum learning rate in adam
K.set_value(vae.optimizer.lr, 1e-4)

####### Callbacks
# path_weights =  #'/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v12/bis4/'
# path_plots = path_weights #'/sps/lsst/users/barcelin/callbacks/LSST/VAE/noisy/v12/bis4/'
# path_tb = path_weights #'/sps/lsst/users/barcelin/Graph/vae_lsst_r_band/noisy/'

alphaChanger = changeAlpha(alpha, vae, vae_loss, path_output)# path_weights)
# Callback to display evolution of training
vae_hist = vae_functions.VAEHistory_2(x_val[:500], vae_utils, latent_dim, alpha, plot_bands=2, figroot=os.path.join(path_output, 'plots/test_noisy_LSST_v4'), period=1)
# Keras Callbacks
#earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0000001, patience=10, verbose=0, mode='min', baseline=None)
checkpointer_mse = ModelCheckpoint(filepath=os.path.join(path_output, 'weights_mse/weights_mse_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt'), monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = ModelCheckpoint(filepath=os.path.join(path_output,'weights_loss/weights_loss_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt'), monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=True, mode='min', period=5)
######## Define all used callbacks
callbacks = [checkpointer_loss, checkpointer_mse, vae_hist, ReduceLROnPlateau(), TerminateOnNaN()]# checkpointer_mse earlystop, checkpointer_loss,vae_hist,, alphaChanger

######## Create generators
list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]

training_generator = BatchGenerator(bands, list_of_samples, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e = None)#180000

validation_generator = BatchGenerator(bands, list_of_samples_val, total_sample_size=None,
                                    batch_size=batch_size, 
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e = None)#180000​
######## Train the network
hist = vae.fit_generator(generator=training_generator, epochs=epochs,
                  steps_per_epoch=32,
                  verbose=1,
                  shuffle=True,
                  validation_data=validation_generator,
                  validation_steps=4,
                  callbacks=callbacks,
                  workers=0)
