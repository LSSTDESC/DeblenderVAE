import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import sys
import os
import logging
import galsim
import cmath as cm
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import WMAP9 as cosmo
import scipy
import scipy.integrate 
from astropy.convolution import Gaussian1DKernel, Gaussian2DKernel, convolve
from cosmos_generation import Gal_generator_noisy_pix_same
from multiprocess import *

# Parameters to fix
phys_stamp_size = 6.4 # arcsec
pixel_scale_euclid_vis = 0.1 # arcsec/pixel

stamp_size = int(phys_stamp_size/pixel_scale_euclid_vis)

# Loading the COSMOS catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample')

 
# function to check if S/N > 2
# Here we do the detection in R band of LSST
def SNR(gal_noiseless,gal_noisy):
    condition = False
    noise = np.std(gal_noisy[6]-gal_noiseless[6])
    max_img_noiseless = np.max(gal_noiseless[6])
    
    snr = abs(max_img_noiseless/noise)
    
    return (snr>2)


import multiprocessing
import time

def map_f(args):
    f, i, v = args
    v.value += 1
    return f(i)
    
def map(func, iter, verbose=True, timesleep=15.0, timeout=None):
    """
    Maps the function func over the iterator iter in a multi-threaded way using the multiprocessing package
    
    func must be pickable, see https://docs.python.org/2/library/pickle.html#what-can-be-pickled-and-unpickled
    
    """
    pool = multiprocessing.Pool()
    m = multiprocessing.Manager()
    v = m.Value(int, 0)
    
    inputs = ((func,i,v) for i in iter) #use a generator, so that nothing is computed before it's needed :)
    
    res = pool.map_async(map_f, inputs)
    
    try :
        n = len(iter)
    except TypeError : # if iter is a generator
        n = None

    if verbose :
        while (True):
            if (res.ready()): break
            # remaining = res._number_left
            # print "Waiting for", remaining, "task chunks to complete..."
            print("# castor.parallel.map : tasks accomplished out of {0} : {1}".format(n, v.get()))
            time.sleep(timesleep)

    pool.close()
    m.shutdown()

    return res.get(timeout)

count = 0
N_cosmo = 2000
N_per_gal = 1

counter = 0

itr = np.arange(N_cosmo)

img_cube_list = []

import time
debut = time.time()

print( 'test')
def func(ind):
    gal_noiseless, gal_noisy, redshift, scale_radius=Gal_generator_noisy_pix_same(cosmos_cat)
    if (SNR(gal_noiseless,gal_noisy) == True):
        return np.array((gal_noiseless,gal_noisy))
    else:
        return func(ind+1) 



img_cube_list = map(func, itr,timesleep = 10.0)# 

# fin = time.time()
# print('time : '+ str(fin-debut))
# i=0

# galaxies = []
# scale_radius_list = []
# SNR_list = []

# while (i < N_cosmo):
#     print(i)
#     gal_noiseless, gal_noisy, redshift, scale_radius=Gal_generator_noisy_pix_same(cosmos_cat)
#     if (SNR(gal_noiseless,gal_noisy) == True):
#         galaxies.append((gal_noiseless,gal_noisy))
#         scale_radius_list.append((scale_radius))
#         SNR_list.append(SNR(gal_noiseless,gal_noisy))
#         i+=1

# fin = time.time()
# print('time : '+ str(fin-debut))

# np.save('/sps/lsst/users/barcelin/data/single/changing_lsst_PSF/independant/galaxies_test_v5', galaxies)
# np.save('/sps/lsst/users/barcelin/data/single/changing_lsst_PSF/independant/scale_radius_test_v5', scale_radius_list)
# np.save('/sps/lsst/users/barcelin/data/single/changing_lsst_PSF/independant/SNR_test_v5', SNR_list)

np.save('/sps/lsst/users/barcelin/data/single/changing_lsst_PSF/independant/galaxies_COSMOS_val_v5_test.npy', img_cube_list)