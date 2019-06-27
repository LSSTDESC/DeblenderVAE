# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import tensorflow.keras
from random import choice
from tensorflow.python.keras.utils import Sequence
from math_functions import sigmoid

#### Generator for LSST R band VAE

class BatchGenerator_lsst_r_band(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST R band only VAE.
    """
    def __init__(self, list_of_samples,total_sample_size, batch_size, training_or_validation, noisy):
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
        self.training_or_validation = training_or_validation
        
        self.x = np.empty([self.batch_size,64,64], dtype='float32')  
        self.y = np.empty([self.batch_size,64,64], dtype='float32')  
        
        self.r = 0

        self.noisy = noisy
        self.step = 0
        self.size = self.batch_size
        self.epoch = 0
        
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
        if (self.training_or_validation == 'training'):
            self.r = np.random.choice(180000-self.batch_size, replace=False)
            if (self.r <=39900):
                self.liste = np.load(self.list_of_samples[0], mmap_mode = 'c')
                self.x = self.liste[self.r:self.r+self.batch_size,1,6]
                self.y = self.liste[self.r:self.r+self.batch_size,0,6]
            else:
                if (self.r <=79900):
                    self.r -= 40000
                    if (self.r >=self.batch_size):
                        self.liste = np.load(self.list_of_samples[1], mmap_mode = 'c')
                        self.x = self.liste[self.r:self.r+self.batch_size,1,6]
                        self.y = self.liste[self.r:self.r+self.batch_size,0,6]
                else:
                    if (self.r <=119900):
                        self.r -= 80000
                        if (self.r >=self.batch_size):
                            self.liste = np.load(self.list_of_samples[2], mmap_mode = 'c')
                            self.x = self.liste[self.r:self.r+self.batch_size,1,6]
                            self.y = self.liste[self.r:self.r+self.batch_size,0,6]
                    else:
                        if (self.r <=159900):
                            self.r -= 120000
                            if (self.r >=self.batch_size):
                                self.liste = np.load(self.list_of_samples[3], mmap_mode = 'c')
                                self.x = self.liste[self.r:self.r+self.batch_size,1,6]
                                self.y = self.liste[self.r:self.r+self.batch_size,0,6]
                        else:
                            self.r -= 180000
                            if (self.r >=self.batch_size):
                                self.liste = np.load(self.list_of_samples[4], mmap_mode = 'c')
                                self.x = self.liste[self.r:self.r+self.batch_size,1,6]
                                self.y = self.liste[self.r:self.r+self.batch_size,0,6]
        # If the generator is a validation generator, only the part dedicated to the validation is displayed
        else:
            self.r = np.random.choice(20000-self.batch_size, replace=False)
            self.liste = np.load(self.list_of_samples[4], mmap_mode = 'c')

            self.x = self.liste[20000+self.r:20000+self.r+self.batch_size,1,6]
            self.y = self.liste[20000+self.r:20000+self.r+self.batch_size,0,6]

        # Preprocessing of the data to be easier for the network to learn
        I= [6.48221069e+05, 4.36202878e+05, 2.27700000e+05, 4.66676013e+04,2.91513800e+02, 2.64974100e+03, 4.66828170e+03, 5.79938030e+03,5.72952590e+03, 3.50687710e+03]
        beta = 5
        for i in range (self.batch_size):
            self.y[i] = np.tanh(np.arcsinh(self.y[i]/(I[6]/beta)))
            self.x[i] = np.tanh(np.arcsinh(self.x[i]/(I[6]/beta)))

        # horizontal flip : flipping the image array of pixels if a random number taken between 0 and 1 is 1
        rand = np.random.randint(2)
        if rand == 1: 
            self.x = np.flipud(self.x)
            self.y = np.flipud(self.y)
        
        indices = np.random.choice(self.batch_size, size=self.size, replace=False)
        for i in indices:
            self.x[i] = self.y[i]

        self.r = 0

        if (self.noisy == True):
            if self.epoch < 0 :
                self.size =self.batch_size
            else:
                #print(self.step, self.epoch)
                if (self.epoch - self.step) > 2:
                    if self.size != 0:
                        self.size -= 20
                        print('new self.size = '+str(self.size))
                    self.step = self.epoch
            self.epoch +=1/(int(float(self.total_sample_size) / float(self.batch_size)))
            
        
        return self.x.reshape((self.batch_size,64,64,1)), self.y.reshape((self.batch_size,64,64,1))




#### Generator for LSST VAE
class BatchGenerator_lsst(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, list_of_samples,total_sample_size, batch_size, training_or_validation, noisy):
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
        self.training_or_validation = training_or_validation
        
        self.x = np.empty([self.batch_size,6,64,64], dtype='float32')  
        self.y = np.empty([self.batch_size,6,64,64], dtype='float32')  
        
        self.r = 0
        
        self.noisy = noisy
        self.step = 0
        self.size = 100
        self.epoch = 0

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
        if (self.training_or_validation == 'training'):
            self.r = np.random.choice(180000-self.batch_size, replace=False)
            if (self.r <=39900):
                self.liste = np.load(self.list_of_samples[0], mmap_mode = 'c')
                self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
            else:
                if (self.r <=79900):
                    self.r -= 40000
                    if (self.r >=100):
                        self.liste = np.load(self.list_of_samples[1], mmap_mode = 'c')
                        self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                        self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                else:
                    if (self.r <=119900):
                        self.r -= 80000
                        if (self.r >=100):
                            self.liste = np.load(self.list_of_samples[2], mmap_mode = 'c')
                            self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                            self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                    else:
                        if (self.r <=159900):
                            self.r -= 120000
                            if (self.r >=100):
                                self.liste = np.load(self.list_of_samples[3], mmap_mode = 'c')
                                self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
                        else:
                            self.r -= 180000
                            if (self.r >=100):
                                self.liste = np.load(self.list_of_samples[4], mmap_mode = 'c')
                                self.x = self.liste[self.r:self.r+self.batch_size,1,4:]
                                self.y = self.liste[self.r:self.r+self.batch_size,0,4:]
        # If the generator is a validation generator, only the part dedicated to the validation is displayed
        else:
            self.r = np.random.choice(20000-self.batch_size, replace=False)
            self.liste = np.load(self.list_of_samples[4], mmap_mode = 'c')

            self.x = self.liste[20000+self.r:20000+self.r+self.batch_size,1,4:]
            self.y = self.liste[20000+self.r:20000+self.r+self.batch_size,0,4:]

        # Preprocessing of the data to be easier for the network to learn
        I= [6.48221069e+05, 4.36202878e+05, 2.27700000e+05, 4.66676013e+04,2.91513800e+02, 2.64974100e+03, 4.66828170e+03, 5.79938030e+03,5.72952590e+03, 3.50687710e+03]
        beta = 5
        for i in range (100):
            for j in range (6):
                self.y[i,j] = np.tanh(np.arcsinh(self.y[i,j]/(I[4+j]/beta)))
                self.x[i,j] = np.tanh(np.arcsinh(self.x[i,j]/(I[4+j]/beta)))

        # horizontal flip : flipping the image array of pixels if a random number taken between 0 and 1 is 1
        rand = np.random.randint(2)
        if rand == 1: 
            self.x = np.flipud(self.x)
            self.y = np.flipud(self.y)
        
        indices = np.random.choice(100, size=self.size, replace=False)
        for i in indices:
            self.x[i] = self.y[i]

        self.r = 0

        if (self.noisy == True):
            if self.epoch < 1 :
                self.size =100
            else:
                if (self.epoch - self.step) > 1:
                    if self.size != 0:
                        self.size -= 20
                        print('new self.size = '+str(self.size))
                    self.step = self.epoch
            self.epoch +=1/(int(float(self.total_sample_size) / float(self.batch_size)))

        return self.x.reshape((self.batch_size,64,64,6)), self.y.reshape((self.batch_size,64,64,6))




#### Generator for LSST + Euclid VAE

class BatchGenerator_lsst_euclid(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST + Euclid VAE.
    """
    def __init__(self, list_of_samples,total_sample_size, batch_size, training_or_validation, noisy):
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
        self.training_or_validation = training_or_validation
        
        self.x = np.empty([self.batch_size,10,64,64], dtype='float32')  
        self.y = np.empty([self.batch_size,10,64,64], dtype='float32')  
        
        self.r = 0
        self.noisy = noisy
        self.step = 0
        self.size = 100
        self.epoch = 0

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
        if (self.training_or_validation == 'training'):
            self.r = np.random.choice(180000-self.batch_size, replace=False)
            if (self.r <=39900):
                self.liste = np.load(self.list_of_samples[0], mmap_mode = 'c')
                self.x = self.liste[self.r:self.r+self.batch_size,1,:]
                self.y = self.liste[self.r:self.r+self.batch_size,0,:]
            else:
                if (self.r <=79900):
                    self.r -= 40000
                    if (self.r >=100):
                        self.liste = np.load(self.list_of_samples[1], mmap_mode = 'c')
                        self.x = self.liste[self.r:self.r+self.batch_size,1,:]
                        self.y = self.liste[self.r:self.r+self.batch_size,0,:]
                else:
                    if (self.r <=119900):
                        self.r -= 80000
                        if (self.r >=100):
                            self.liste = np.load(self.list_of_samples[2], mmap_mode = 'c')
                            self.x = self.liste[self.r:self.r+self.batch_size,1,:]
                            self.y = self.liste[self.r:self.r+self.batch_size,0,:]
                    else:
                        if (self.r <=159900):
                            self.r -= 120000
                            if (self.r >=100):
                                self.liste = np.load(self.list_of_samples[3], mmap_mode = 'c')
                                self.x = self.liste[self.r:self.r+self.batch_size,1,:]
                                self.y = self.liste[self.r:self.r+self.batch_size,0,:]
                        else:
                            self.r -= 180000
                            if (self.r >=100):
                                self.liste = np.load(self.list_of_samples[4], mmap_mode = 'c')
                                self.x = self.liste[self.r:self.r+self.batch_size,1,:]
                                self.y = self.liste[self.r:self.r+self.batch_size,0,:]
        # If the generator is a validation generator, only the part dedicated to the validation is displayed
        else:
            self.r = np.random.choice(20000-self.batch_size, replace=False)
            self.liste = np.load(self.list_of_samples[4], mmap_mode = 'c')

            self.x = self.liste[20000+self.r:20000+self.r+self.batch_size,1,:]
            self.y = self.liste[20000+self.r:20000+self.r+self.batch_size,0,:]

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

        
        indices = np.random.choice(100, size=self.size, replace=False)
        for i in indices:
            self.x[i] = self.y[i]

        self.r = 0

        if (self.noisy == True):
            if self.epoch < 1 :
                self.size =100
            else:
                if (self.epoch - self.step) > 1:
                    if self.size != 0:
                        self.size -= 5
                        print('new self.size = '+str(self.size))
                    self.step = self.epoch
            self.epoch +=1/(int(float(self.total_sample_size) / float(self.batch_size)))


        return self.x.reshape((self.batch_size,64,64,10)), self.y.reshape((self.batch_size,64,64,10))




