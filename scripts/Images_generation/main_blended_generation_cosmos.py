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
from blended_images import Gal_generator_noisy_test, Gal_generator_noisy_training
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
N_cosmo = 10000
N_per_gal = 5

ud = galsim.UniformDeviate()
rng = galsim.BaseDeviate()

counter = 0

itr = np.arange(N_cosmo)

img_cube_list = []

import time



# def func(ind):
#     nb_blended_gal = np.random.randint(1,5)
#     galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy,shift, redshift = Gal_generator_noisy_test(cosmos_cat, nb_blended_gal)
#     if (SNR(galaxy_noiseless, blend_noisy) == True):
#         return np.array((galaxy_noiseless, blend_noisy)), np.array((shift, redshift))
#     else:
#         return func(ind+1) 

# debut = time.time()

# img_cube_list = map(func, itr,timesleep = 10.0)# 
# print(len(img_cube_list, len(img_cube_list[0])))

# fin = time.time()
# print('time : '+ str(fin-debut))

#np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_test_2_v4.npy', img_cube_list)


i = 0
galaxies = []
shift_list = []
redshift_list  = []
magnitude_list = []

while (i < N_cosmo):
    print(i)
    nb_blended_gal = np.random.randint(1,N_per_gal)
    galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy,shift, redshift, mag = Gal_generator_noisy_test(cosmos_cat, nb_blended_gal)
    if (SNR(galaxy_noiseless, blend_noisy) == True):
        galaxies.append((galaxy_noiseless, blend_noisy))
        shift_list.append((shift))
        redshift_list.append((redshift))
        magnitude_list.append((mag))
        i+=1

#     fin = time.time()
#     print('time : '+ str(fin-debut))

np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_test_2_v5.npy', galaxies)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_test_shift_v5.npy', shift_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_test_redshift_v5.npy', redshift_list)
np.save('/sps/lsst/users/barcelin/data/blended/COSMOS/galaxies_COSMOS_test_magnitude_v5.npy', magnitude_list)