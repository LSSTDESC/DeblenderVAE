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
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import tensorflow as tf
import tensorflow_probability as tfp

# from generator_deblender import BatchGenerator
sys.path.insert(0,'../VAE/')

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions, utils, generator

######## Set some parameters
batch_size = 100
latent_dim = 32
epochs = int(sys.argv[4])#10000
bands = [0,1,2,3,4,5,6,7,8,9]

steps_per_epoch = 32 #256
validation_steps = 8 #16

load_from_vae_or_deblender = str(sys.argv[5])#'vae'

images_dir = '/sps/lsst/users/barcelin/data/blended_galaxies/'+str(sys.argv[3])+'validation/'
path_output = '/sps/lsst/users/barcelin/weights/LSST_EUCLID/deblender/'+str(sys.argv[6])
path_output_vae = '/sps/lsst/users/barcelin/weights/LSST_EUCLID/VAE/noisy/'+str(sys.argv[6])#v13/bis


######## Import data for callback (Only if VAEHistory is used)
x_val = np.load(os.path.join(images_dir, 'galaxies_blended_20191024_0_images.npy'))[:500,:,bands].transpose([0,1,3,4,2])


# ####### Load deblender
if load_from_vae_or_deblender == 'vae':
    deblender, deblender_utils, encoder, decoder, Dkl = utils.load_vae_full(path_output_vae, 10, folder=True) 
elif load_from_vae_or_deblender == 'deblender':
    deblender, deblender_utils, encoder, decoder, Dkl = utils.load_vae_full(path_output, 10, folder=True) 
else:
    raise NotImplementedError
decoder.trainable = True
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
K.set_value(deblender.optimizer.lr, sys.argv[1])

#######
# Callback
path_weights = '/sps/lsst/users/barcelin/weights/LSST_EUCLID/deblender/'+str(sys.argv[2])#v8/bis/#v7/bis_bis/#6/bis_bis
path_plots = '/sps/lsst/users/barcelin/callbacks/LSST_EUCLID/deblender/'+str(sys.argv[2])#v8/bis/#v7/bis_bis/#6/bis_bis
#path_tb = '/sps/lsst/users/barcelin/Graph/deblender_lsst_euclid/'

#tbCallBack = tf.keras.callbacks.TensorBoard(log_dir=path_tb+'noiseless/', histogram_freq=0, batch_size = batch_size, write_graph=True, write_images=True)
vae_hist = vae_functions.VAEHistory(x_val[:500], deblender_utils, latent_dim, alpha, plot_bands=[1,2,3], figroot=os.path.join(path_plots, 'test_noisy_LSST_v4'), period=2)
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath=path_weights+'loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)

######## Define all used callbacks
callbacks = [checkpointer_mse, vae_hist, checkpointer_loss]
 
######## List of data samples
# list_of_samples=['/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_1_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_2_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_3_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_4_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_5_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_6_v5.npy',
#                 '/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_7_v5.npy']


# list_of_samples_val=['/sps/lsst/users/barcelin/data/blended/COSMOS/PSF_lsst_0.65/uni11/galaxies_blended_val_v5.npy']

images_dir = '/sps/lsst/users/barcelin/data/blended_galaxies/'+str(sys.argv[3])
images_dir_real = '/sps/lsst/users/barcelin/data/blended_galaxies/real'
list_of_samples = [x for x in utils.listdir_fullpath(os.path.join(images_dir,'training')) if x.endswith('.npy')]+[x for x in utils.listdir_fullpath(os.path.join(images_dir_real,'training')) if x.endswith('.npy')]
list_of_samples_val =[x for x in utils.listdir_fullpath(os.path.join(images_dir_real,'validation')) if x.endswith('.npy')]# [x for x in utils.listdir_fullpath(os.path.join(images_dir,'validation')) if x.endswith('.npy')]+


######## Define the generators
training_generator = generator.BatchGenerator(bands, list_of_samples,total_sample_size=None,
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
hist = deblender.fit_generator(training_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
        shuffle = True,
        validation_steps =validation_steps,
        validation_data=validation_generator, 
        callbacks=callbacks,
        workers=0, 
        use_multiprocessing = True)

