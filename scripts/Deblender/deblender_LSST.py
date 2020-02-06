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

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions, utils, generator
from tools_for_VAE.callbacks import changeAlpha

######## Set some parameters
batch_size = 100
latent_dim = 32
epochs = int(sys.argv[4])#10000
bands = [4,5,6,7,8,9]

steps_per_epoch = 128 #256
validation_steps = 16 #16

load_from_vae_or_deblender = str(sys.argv[5])#'deblender'

images_dir = '/sps/lsst/users/barcelin/data/blended_galaxies/'+str(sys.argv[3])+'validation/'
path_output = '/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/'+str(sys.argv[6])#7/mse
path_output_vae = '/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/'+str(sys.argv[6])

######## Import data for callback (Only if VAEHistory is used)
x_val = np.load(os.path.join(images_dir, 'galaxies_blended_20191024_0_images.npy'))[:500,:,bands].transpose([0,1,3,4,2])


######## Load deblender
if load_from_vae_or_deblender == 'vae':
    deblender, deblender_utils, encoder, decoder, Dkl = utils.load_vae_full(path_output_vae, 6, folder=True) 
elif load_from_vae_or_deblender == 'deblender':
    deblender, deblender_utils, encoder, decoder, Dkl = utils.load_vae_full(path_output, 6, folder=True) 
else:
    raise NotImplementedError
decoder.trainable = False

# Define the loss function
alpha = K.variable(1e-2)

def deblender_loss(x, x_decoded_mean):
    xent_loss = K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
    kl_loss = K.get_value(alpha) * Dkl
    return xent_loss + K.mean(kl_loss)

######## Compile the VAE
deblender.compile('adam', loss=deblender_loss, metrics=['mse'])
print(deblender.summary())
######## Fix the maximum learning rate in adam
K.set_value(deblender.optimizer.lr, sys.argv[1])

#######
# Callback

path_weights = '/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/'+str(sys.argv[2])
path_plots = '/sps/lsst/users/barcelin/callbacks/LSST/deblender/noisy/'+str(sys.argv[2])
path_tb = '/sps/lsst/users/barcelin/Graph/deblender_lsst/'

# alphaChanger = callbacks.changeAlpha(alpha, deblender, deblender_loss, path_weights)
# ######## Define all used callbacks
# callbacks = [vae_hist,checkpointer_mse]#vae_hist, , checkpointer_loss

#alphaChanger = changeAlpha(alpha, deblender, deblender_loss, path_output)# path_weights)
# Callback to display evolution of training
vae_hist = vae_functions.VAEHistory(x_val[:500], deblender_utils, latent_dim, alpha, plot_bands=[1,2,3], figroot=os.path.join(path_plots, 'test_noisy_LSST_v4'), period=2)
# Keras Callbacks
#earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_mean_squared_error', min_delta=0.0000001, patience=10, verbose=0, mode='min', baseline=None)
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

######## Define all used callbacks
callbacks = [checkpointer_mse, vae_hist, checkpointer_loss]
 
######## List of data samples
images_dir = '/sps/lsst/users/barcelin/data/blended_galaxies/'+str(sys.argv[3])
list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]
list_of_samples_val = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]

######## Define the generators
training_generator = generator.BatchGenerator(bands, list_of_samples, total_sample_size=None,
                                    batch_size=batch_size,
                                    trainval_or_test='training',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e = None)#180000

validation_generator = generator.BatchGenerator(bands, list_of_samples_val, total_sample_size=None,
                                    batch_size=batch_size,
                                    trainval_or_test='validation',
                                    do_norm=False,
                                    denorm = False,
                                    list_of_weights_e = None)#180000

######## Train the network
hist = deblender.fit_generator(generator=training_generator, epochs=epochs,
                  steps_per_epoch=steps_per_epoch,
                  verbose=2,
                  shuffle=True,
                  validation_data=validation_generator,
                  validation_steps=validation_steps,
                  callbacks=callbacks,
                  workers=0, 
                  use_multiprocessing = True)
