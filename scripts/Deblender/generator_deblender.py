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
    def __init__(self, bands, list_of_samples,total_sample_size, batch_size, size_of_lists, training_or_validation, noisy):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.training_or_validation = training_or_validation 
        
        self.r = 0
        
        self.noisy = noisy
        self.step = 0
        self.size = 100
        self.epoch = 0

        self.size_of_lists = size_of_lists

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

        #print(self.bands, self.r,)
        #print(self.liste[self.r, 1][:,[6]].shape)
        #print(self.liste[self.r,1,self.bands].shape, self.liste[self.r,1,self.bands])
        self.x = self.liste[self.r,1][:,self.bands]
        self.y = self.liste[self.r,0][:,self.bands]
        
        #if len(self.bands) == 1:
        #    self.x = np.expand_dims(self.x, axis = 1)
        #    self.y = np.expand_dims(self.y, axis = 1)

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
        
        return self.x, self.y






class BatchGenerator_test(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands, list_of_samples,total_sample_size, batch_size, magnitude, shift, training_or_validation, noisy):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.bands = bands
        self.nbands = len(bands)
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
        self.training_or_validation = training_or_validation 
        
        self.r = 0
        
        self.noisy = noisy
        self.step = 0
        self.size = 100
        self.epoch = 0

        self.magnitude = magnitude
        self.shift = shift

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

        self.mag = self.magnitude[self.r]
        self.s = self.shift[self.r]
        
        # Preprocessing of the data to be easier for the network to learn
        self.x = utils.norm(self.x, self.bands)
        self.y = utils.norm(self.y, self.bands)
        
        self.x = np.transpose(self.x, axes = (0,2,3,1))
        self.y = np.transpose(self.y, axes = (0,2,3,1))
        
        return self.x, self.y, self.mag, self.s






#### Generator for LSST deblender







class BatchGenerator_lsst_process(tf.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST deblender
    """
    def __init__(self, list_of_samples,total_sample_size, batch_size,size_of_lists, training_or_validation):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
#        self.path = random.choice(self.list_of_samples)
#        print(self.path)
        self.training_or_validation = training_or_validation
        
        self.x = np.empty([self.batch_size,6,64,64], dtype='float32')  
        self.y = np.empty([self.batch_size,6,64,64], dtype='float32')  
        
#        self.liste = np.load(self.path, mmap_mode = 'c')
        self.r = 0
        self.size_of_lists = size_of_lists
        
    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
#        path = random.choice(self.list_of_samples)
#        print(path)
#        self.liste = np.load(path, mmap_mode = 'c')
        self.r = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        if (self.training_or_validation == 'training'):
            self.r = np.random.choice(self.total_sample_size-self.batch_size, replace=False)
            if (self.r <=19900):
                self.liste = np.load(self.list_of_samples[0], mmap_mode = 'c')
                self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
            else:
                if (self.r <=39900):
                    self.r -= 20000
                    if (self.r >=100):
                        self.liste = np.load(self.list_of_samples[1], mmap_mode = 'c')
                        self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                        self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                else:
                    if (self.r <=59900):
                        self.r -= 40000
                        if (self.r >=100):
                            self.liste = np.load(self.list_of_samples[2], mmap_mode = 'c')
                            self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                            self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                    else:
                        if (self.r <=79900):
                            self.r -= 60000
                            if (self.r >=100):
                                self.liste = np.load(self.list_of_samples[3], mmap_mode = 'c')
                                self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                        else:
                            if (self.r <=99900):
                                self.r -= 80000
                                if (self.r >=100):
                                    self.liste = np.load(self.list_of_samples[4], mmap_mode = 'c')
                                    self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                    self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                            else:
                                if (self.r <=119900):
                                    self.r -= 100000
                                    if (self.r >=100):
                                        self.liste = np.load(self.list_of_samples[5], mmap_mode = 'c')
                                        self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                        self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                                else:
                                    if (self.r <=139900):
                                        self.r -= 120000
                                        if (self.r >=100):
                                            self.liste = np.load(self.list_of_samples[6], mmap_mode = 'c')
                                            self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                            self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                                    else:
                                        if (self.r <=159900):
                                            self.r -= 140000
                                            if (self.r >=100):
                                                self.liste = np.load(self.list_of_samples[7], mmap_mode = 'c')
                                                self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                                self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                                        else:
                                            if (self.r <=179900):
                                                self.r -= 160000
                                                if (self.r >=100):
                                                    self.liste = np.load(self.list_of_samples[8], mmap_mode = 'c')
                                                    self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                                    self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                                            else:
                                                if (self.r <=189900):
                                                    self.r -= 180000
                                                    if (self.r >=100):
                                                        self.liste = np.load(self.list_of_samples[9], mmap_mode = 'c')
                                                        self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                                        self.y = self.liste[self.r:self.r+self.batch_size,0,4:]

        # If the generator is a validation generator, only the part dedicated to the validation is displayed
        else:
            self.r = np.random.choice(self.total_sample_size-self.batch_size, replace=False)
            if self.training_or_validation == "validation":
                self.liste = np.load(self.list_of_samples[9], mmap_mode = 'c')
            else: 
                self.liste = np.load(self.list_of_samples[0], mmap_mode = 'c')
            self.x = self.liste[self.size_of_lists - self.total_sample_size+self.r:self.size_of_lists - self.total_sample_size+self.r+self.batch_size,1,4:]
            self.y = self.liste[self.size_of_lists - self.total_sample_size+self.r:self.size_of_lists - self.total_sample_size+self.r+self.batch_size,0,4:]
        # Preprocessing of the data to be easier for the network to learn
        I= [6.48221069e+05, 4.36202878e+05, 2.27700000e+05, 4.66676013e+04,2.91513800e+02, 2.64974100e+03, 4.66828170e+03, 5.79938030e+03,5.72952590e+03, 3.50687710e+03]
        beta = 5
        for i in range (100):
            for j in range (6):
                self.y[i,j] = np.tanh(np.arcsinh(self.y[i,j]/(I[j+4]/beta)))
                self.x[i,j] = np.tanh(np.arcsinh(self.x[i,j]/(I[j+4]/beta)))

        # horizontal flip : flipping the image array of pixels if a random number taken between 0 and 1 is 1
        rand = np.random.randint(2)
        if rand == 1: 
            self.x = np.flipud(self.x)
            self.y = np.flipud(self.y)
        
        return self.x.reshape((self.batch_size,64,64,6)), self.y.reshape((self.batch_size,64,64,6))





class BatchGenerator_lsst_euclid_process(tf.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST deblender
    """
    def __init__(self, list_of_samples,total_sample_size, batch_size,size_of_lists, training_or_validation):
        """
        Initialization function
        total_sample_size: size of the whole training (or validation) sample
        batch_size: size of the batches to provide
        list_of_samples: list of the numpy arrays which correspond to the whole training (or validation) sample
#        path: path to the first numpy array taken in which the batch will be taken
        training_or_validation: choice between training of validation generator
        x: input of the neural network
        y: target of the neural network
        r: random value to sample into the validation sample
        """
        self.total_sample_size = total_sample_size
        self.batch_size = batch_size
        self.list_of_samples = list_of_samples
#        self.path = random.choice(self.list_of_samples)
#        print(self.path)
        self.training_or_validation = training_or_validation
        
        self.x = np.empty([self.batch_size,10,64,64], dtype='float32')  
        self.y = np.empty([self.batch_size,10,64,64], dtype='float32')  
        
#        self.liste = np.load(self.path, mmap_mode = 'c')
        self.r = 0
        self.size_of_lists = size_of_lists
        
    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
#        path = random.choice(self.list_of_samples)
#        print(path)
#        self.liste = np.load(path, mmap_mode = 'c')
        self.r = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        if (self.training_or_validation == 'training'):
            self.r = np.random.choice(self.total_sample_size-self.batch_size, replace=False)
            if (self.r <=19900):
                self.liste = np.load(self.list_of_samples[0], mmap_mode = 'c')
                self.x = self.liste[self.r:self.r+self.batch_size,1]
                self.y = self.liste[self.r:self.r+self.batch_size,0]
            else:
                if (self.r <=39900):
                    self.r -= 20000
                    if (self.r >=100):
                        self.liste = np.load(self.list_of_samples[1], mmap_mode = 'c')
                        self.x = self.liste[self.r:self.r+self.batch_size,1]
                        self.y = self.liste[self.r:self.r+self.batch_size,0]
                else:
                    if (self.r <=59900):
                        self.r -= 40000
                        if (self.r >=100):
                            self.liste = np.load(self.list_of_samples[2], mmap_mode = 'c')
                            self.x = self.liste[self.r:self.r+self.batch_size,1]
                            self.y = self.liste[self.r:self.r+self.batch_size,0]
                    else:
                        if (self.r <=79900):
                            self.r -= 60000
                            if (self.r >=100):
                                self.liste = np.load(self.list_of_samples[3], mmap_mode = 'c')
                                self.x = self.liste[self.r:self.r+self.batch_size,1]
                                self.y = self.liste[self.r:self.r+self.batch_size,0]
                        else:
                            if (self.r <=99900):
                                self.r -= 80000
                                if (self.r >=100):
                                    self.liste = np.load(self.list_of_samples[4], mmap_mode = 'c')
                                    self.x = self.liste[self.r:self.r+self.batch_size,1]
                                    self.y = self.liste[self.r:self.r+self.batch_size,0]
                            else:
                                if (self.r <=119900):
                                    self.r -= 100000
                                    if (self.r >=100):
                                        self.liste = np.load(self.list_of_samples[5], mmap_mode = 'c')
                                        self.x = self.liste[self.r:self.r+self.batch_size,1]
                                        self.y = self.liste[self.r:self.r+self.batch_size,0]
                                else:
                                    if (self.r <=139900):
                                        self.r -= 120000
                                        if (self.r >=100):
                                            self.liste = np.load(self.list_of_samples[6], mmap_mode = 'c')
                                            self.x = self.liste[self.r:self.r+self.batch_size,1]
                                            self.y = self.liste[self.r:self.r+self.batch_size,0]
                                    else:
                                        if (self.r <=159900):
                                            self.r -= 140000
                                            if (self.r >=100):
                                                self.liste = np.load(self.list_of_samples[7], mmap_mode = 'c')
                                                self.x = self.liste[self.r:self.r+self.batch_size,1]
                                                self.y = self.liste[self.r:self.r+self.batch_size,0]
                                        else:
                                            if (self.r <=179900):
                                                self.r -= 160000
                                                if (self.r >=100):
                                                    self.liste = np.load(self.list_of_samples[8], mmap_mode = 'c')
                                                    self.x = self.liste[self.r:self.r+self.batch_size,1]
                                                    self.y = self.liste[self.r:self.r+self.batch_size,0]
                                            else:
                                                if (self.r <=189900):
                                                    self.r -= 180000
                                                    if (self.r >=100):
                                                        self.liste = np.load(self.list_of_samples[9], mmap_mode = 'c')
                                                        self.x = self.liste[self.r:self.r+self.batch_size,1]
                                                        self.y = self.liste[self.r:self.r+self.batch_size,0]

        # If the generator is a validation generator, only the part dedicated to the validation is displayed
        else:
            self.r = np.random.choice(self.total_sample_size-self.batch_size, replace=False)
            if self.training_or_validation == "validation":
                self.liste = np.load(self.list_of_samples[9], mmap_mode = 'c')
            else: 
                self.liste = np.load(self.list_of_samples[0], mmap_mode = 'c')

            self.x = self.liste[self.size_of_lists - self.total_sample_size+self.r:self.size_of_lists - self.total_sample_size+self.r+self.batch_size,1]
            self.y = self.liste[self.size_of_lists - self.total_sample_size+self.r:self.size_of_lists - self.total_sample_size+self.r+self.batch_size,0]

        # Preprocessing of the data to be easier for the network to learn
        I= [6.48221069e+05, 4.36202878e+05, 2.27700000e+05, 4.66676013e+04,2.91513800e+02, 2.64974100e+03, 4.66828170e+03, 5.79938030e+03,5.72952590e+03, 3.50687710e+03]
        beta = 5
        for i in range (100):
            for j in range (10):
                self.y[i,j] = np.tanh(np.arcsinh(self.y[i,j]/(I[j]/beta)))
                self.x[i,j] = np.tanh(np.arcsinh(self.x[i,j]/(I[j]/beta)))

        # horizontal flip : flipping the image array of pixels if a random number taken between 0 and 1 is 1
        rand = np.random.randint(2)
        if rand == 1: 
            self.x = np.flipud(self.x)
            self.y = np.flipud(self.y)
        
        return self.y.reshape((self.batch_size,64,64,10)), self.y.reshape((self.batch_size,64,64,10))
