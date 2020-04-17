import sys, os
import numpy as np
import matplotlib.pyplot as plt


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Dropout, MaxPool2D, Flatten, BatchNormalization, Reshape, UpSampling2D, Cropping2D, Conv2DTranspose, PReLU, Concatenate, Lambda
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, TerminateOnNaN, ModelCheckpoint
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras import metrics


###### Callbacks
# Create a callback for changing KL coefficient in the loss
class changeAlpha(Callback):
    def __init__(self, alpha, vae, vae_loss, path):
        '''
        Initialization function
        epoch: epoch number, initialized at 1
        alpha: value of the KL coefficient
        vae: network trained
        vae_loss: loss used to train the vae
        path: savepath for the value of alpha
        '''
        self.epoch = 1
        self.alpha = alpha
        self.vae = vae
        self.vae_loss = vae_loss
        self.path = path
    
    def on_epoch_end(self, alpha, vae):
        '''
        Function to change alpha at the end of the epoch
        '''
        # Alpha is stable for the ten first epochs
        stable = 10
        # During these first epoch alpha is fixed at 0.0001
        new_alpha = 0.0001
        # Change alpha only when epoch > 10 and alpha > 1.e-8
        if self.epoch > stable and K.get_value(self.alpha)>1e-8 :
            # The change is done by dividing alpha by 2
            new_alpha = K.get_value(self.alpha)/2
            K.set_value(self.alpha, new_alpha)
            self.vae.compile('adam', loss=self.vae_loss, metrics=['mse'])
            print('loss modified')
            self.epoch = 1
            # Save the last used value of alpha
            np.save(self.path+'alpha', K.get_value(self.alpha))
        
        self.epoch +=1

# Create a callback to change learning rate of the optimizer during training
class changelr(Callback):
    def __init__(self,vae):
        '''
        Function to change learning rate during training
        '''
        self.epoch = 0
        self.vae = vae
    
    def on_epoch_end(self, alpha, vae):
        # Change learning rate only after 100th epoch
        if (self.epoch == 100):
            self.epoch =0
            actual_value = K.get_value(self.vae.optimizer.lr)
            # Change the learning rate only while lr > 0.000009
            if (actual_value > 0.000009):
                new_value = actual_value/2
                K.set_value(self.vae.optimizer.lr, new_value)
                print(K.get_value(self.vae.optimizer.lr))
        self.epoch +=1
