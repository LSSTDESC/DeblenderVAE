# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import tensorflow.keras
from random import choice
from tensorflow.python.keras.utils import Sequence

import tensorflow as tf
sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import utils

class BatchGenerator(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands, list_of_samples,total_sample_size, batch_size, magnitude, shift, blendedness, scale_radius, trainval_or_test, noisy):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        trainval_or_test : choice between training/validation generator or test generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.trainval_or_test = trainval_or_test 
        
        self.r = 0
        
        self.noisy = noisy
        self.step = 0
        self.size = 100
        self.epoch = 0

        self.magnitude = magnitude
        self.shift = shift
        self.blendedness = blendedness
        print(self.blendedness.shape)
        self.scale_radius = scale_radius
        print(self.scale_radius.shape)

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.p /= np.sum(self.p)

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        self.r = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        self.liste = np.load(np.random.choice(self.list_of_samples, p = self.p), mmap_mode = 'c')
        self.r = np.random.choice(len(self.liste), size = self.batch_size, replace=False)

        self.x = self.liste[self.r,1][:,self.bands]
        self.y = self.liste[self.r,0][:,self.bands]

        # Preprocessing of the data to be easier for the network to learn
        self.x = utils.norm(self.x, self.bands)
        self.y = utils.norm(self.y, self.bands)

        #  flip : flipping the image array
        rand = np.random.randint(4)
        if rand == 1: 
            self.x = np.flip(self.x, axis = -1)
            self.y = np.flip(self.y, axis = -1)
        elif rand == 2 : 
            self.x = np.flip(self.x, axis = -2)
            self.y = np.flip(self.y, axis = -2)
        elif rand == 3:
            self.x = np.flip(self.x, axis = (-1,-2))
            self.y = np.flip(self.y, axis = (-1,-2))
        
        self.x = np.transpose(self.x, axes = (0,2,3,1))
        self.y = np.transpose(self.y, axes = (0,2,3,1))
        
        if self.trainval_or_test == 'trainval':
            return self.x, self.y
        elif self.trainval_or_test == 'test':
            self.mag = self.magnitude[self.r]
            self.s = self.shift[self.r]
            self.blend = self.blendedness[self.r]
            self.radius = self.scale_radius[self.r]

            self.delta_r, self.delta_mag, self.blend_max = utils.compute_deltas_for_most_blended(self.s,self.mag,self.blend)
            return self.x, self.y, self.mag, self.s, self.delta_r, self.delta_mag, self.blend_max, self.blend, self.radius
