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

from generator_deblender import BatchGenerator_lsst_process

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import model, vae_functions, utils


# Fix some Parameters
batch_size = 100
original_dim = 64*64*6
epochs = 1000
epsilon_std = 1.0

input_shape = (64,64,6)
latent_dim = 10
hidden_dim = 256
filters = [32,64, 128, 256]
kernels = [3,3,3,3]

# Test data for callback
x = np.load('/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_1_v4.npy')
x_val = x[:500,1,4:]

I= [6.48221069e+05, 4.36202878e+05, 2.27700000e+05, 4.66676013e+04,2.91513800e+02, 2.64974100e+03, 4.66828170e+03, 5.79938030e+03,5.72952590e+03, 3.50687710e+03]
beta = 5
for i in range (500):
    for j in range (6):
        x_val[i,j] = np.tanh(np.arcsinh(x_val[i,j]/(I[j+4]/beta)))
x_val = x_val.reshape((len(x_val),64,64,6))

# Load decoder of VAE
decoder = utils.load_vae_decoder('/sps/lsst/users/barcelin/weights/LSST/VAE/noisy/v5/mse/',6,folder = True)
decoder.trainable = False

# Deblender model
deb_encoder, deb_decoder = model.vae_model(6)

# Use the encoder of the trained VAE
deblender, deblender_utils, output_encoder_deb = vae_functions.build_vanilla_vae(deb_encoder, decoder, full_cov=False, coeff_KL = 0)


# Define the loss function
alpha = K.variable(0.0001)

def deblender_loss(x, x_decoded_mean):
    xent_loss = original_dim*K.mean(K.sum(K.binary_crossentropy(x, x_decoded_mean), axis=[1,2,3]))
    #kl_loss = - .5 * K.get_value(alpha) * K.sum(1 + output_encoder_deb[1] - K.square(output_encoder_deb[0]) - K.exp(output_encoder_deb[1]), axis=-1)
    return xent_loss #+ K.mean(kl_loss))


deblender.compile('adam', loss=deblender_loss, metrics=['mse'])
K.set_value(deblender.optimizer.lr, 0.0001)
###### Fix some parameters
phys_stamp_size = 6.4 # arcsec
pixel_scale_euclid_vis = 0.1 # arcsec/pixel

stamp_size = int(phys_stamp_size/pixel_scale_euclid_vis)
#######
#######
# Callback
tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='/sps/lsst/users/barcelin/Graph/deblender_lsst/noiseless/', histogram_freq=0, batch_size = batch_size, write_graph=True, write_images=True)

#vae_hist = vae_functions.VAEHistory(x_val[:500], deblender_utils, latent_dim, alpha, plot_bands=[3,4,5], figname='/sps/lsst/users/barcelin/callbacks/LSST/deblender/noisy/v1/test_')
checkpointer_mse = tf.keras.callbacks.ModelCheckpoint(filepath='/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/v2/mse/weights_noisy_v4.{epoch:02d}-{val_mean_squared_error:.2f}.ckpt', monitor='val_mean_squared_error', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
checkpointer_loss = tf.keras.callbacks.ModelCheckpoint(filepath='/sps/lsst/users/barcelin/weights/LSST/deblender/noisy/v2/loss/weights_noisy_v4.{epoch:02d}-{val_loss:.2f}.ckpt', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True, mode='min', period=1)
callbacks = [checkpointer_mse, checkpointer_loss]#vae_hist, 
 
##### If processing with generator (apply Cyrille's function)
list_of_samples=['/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_1_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_2_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_3_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_4_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_5_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_6_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_7_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_8_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_9_v4.npy',
'/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_10_v4.npy']



training_generator = BatchGenerator_lsst_process(list_of_samples,total_sample_size=10000, batch_size= batch_size, size_of_lists = 20000, training_or_validation = 'training')#190000
validation_generator = BatchGenerator_lsst_process(list_of_samples,total_sample_size=300, batch_size= batch_size, size_of_lists = 20000, training_or_validation = 'validation')#10000


hist = deblender.fit_generator(training_generator,
        epochs=epochs,
        steps_per_epoch=100,#1900,
        verbose=2,
        shuffle = True,
        validation_steps =3,#100,
        validation_data=validation_generator,
        callbacks=callbacks,
        workers = 0)#,)#, callbacks = [tbCallBack ])#,plot_learning

# plt.plot(hist.history['mean_squared_error'])
# plt.plot(hist.history['val_mean_squared_error'])
# plt.title('model loss')
# plt.ylabel('mse')
# plt.xlabel('epoch')

# plt.legend(['train', 'test'], loc='upper left')
#plt.show()
#plt.savefig('loss_lsst_conv_deblender_noisy_Normy.png')

# deblender.save("/sps/lsst/users/barcelin/deblender-LSST_conv_noiseless_test_v1.hdf5") 
#deblender.save_weights("/pbs/throng/lsst/users/barcelin/deblender_LSST/test")