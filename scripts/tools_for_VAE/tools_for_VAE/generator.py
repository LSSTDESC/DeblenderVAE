# Import necessary librairies

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random
import tensorflow.keras
import pandas as pd
from random import choice
from tensorflow.python.keras.utils import Sequence

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import utils



class BatchGenerator(tensorflow.keras.utils.Sequence):
    """
    Class to create batch generator for the LSST VAE.
    """
    def __init__(self, bands, list_of_samples,total_sample_size, batch_size, trainval_or_test, do_norm, list_of_weights_e):
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
        self.trainval_or_test = trainval_or_test
        
        # indices = 0
        
        #self.noisy = noisy
        # self.step = 0
        # self.size = 100
        self.epoch = 0
        self.do_norm = do_norm

        # self.scale_radius = scale_radius
        # self.SNR = SNR

        # Weights computed from the lengths of lists
        self.p = []
        for sample in self.list_of_samples:
            temp = np.load(sample, mmap_mode = 'c')
            self.p.append(float(len(temp)))
        self.p = np.array(self.p)
        self.total_sample_size = int(np.sum(self.p))
        print("[BatchGenerator] total_sample_size = ", self.total_sample_size)
        print("[BatchGenerator] len(list_of_samples) = ", len(self.list_of_samples))

        self.p /= np.sum(self.p)

        self.produced_samples = 0
        self.list_of_weights_e = list_of_weights_e
        #self.shifts = shifts

    def __len__(self):
        """
        Function to define the length of an epoch
        """
        return int(float(self.total_sample_size) / float(self.batch_size))      

    def on_epoch_end(self):
        """
        Function executed at the end of each epoch
        """
        # indices = 0
        print("Produced samples", self.produced_samples)
        self.produced_samples = 0
        
    def __getitem__(self, idx):
        """
        Function which returns the input and target batches for the network
        """
        # If the generator is a training generator, the whole sample is displayed
        #sample_filename = np.random.choice(self.list_of_samples, p=self.p)
        index = np.random.choice(list(range(len(self.p))), p=self.p)
        sample_filename = self.list_of_samples[index]
        sample = np.load(sample_filename, mmap_mode = 'c')

        if self.list_of_weights_e == None:
            indices = np.random.choice(len(sample), size=self.batch_size, replace=False)
        else:
            self.weights_e = np.load(self.list_of_weights_e[index])[:,0]
            indices = np.random.choice(len(sample), size=self.batch_size, replace=False, p = self.weights_e/np.sum(self.weights_e))
            print(indices)
        self.produced_samples += len(indices)
        
        self.x = sample[indices,1][:,self.bands]
        self.y = sample[indices,0][:,self.bands]
        
        # Preprocessing of the data to be easier for the network to learn
        if self.do_norm:
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
        
        if self.trainval_or_test == 'training' or self.trainval_or_test == 'validation':
            return self.x, self.y
        elif self.trainval_or_test == 'test':
            # indicesadius = self.scale_radius[indices]
            # self.SNR_out = self.SNR[indices]
            data = pd.read_csv(sample_filename.replace('images.npy','data.csv'))
            #if self.shifts != None:
            #   return self.x, self.y, data.loc[indices], self.shifts[indices], indices
            #else:
            return self.x, self.y, data.loc[indices], indices