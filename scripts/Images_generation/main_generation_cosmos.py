# Import packages

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import logging
import galsim
import cmath as cm
import matplotlib.pyplot as plt
from multiprocess import *
import pandas as pd
from tqdm import tqdm, trange

sys.path.insert(0,'../tools_for_VAE/')
import tools_for_VAE
from tools_for_VAE import utils

from images_generator import image_generator, image_generator_real

# The script is used as, eg,
# # python main_blended_generation_cosmos.py training 10 1000
# to produce 10 files in the training sample with 1000 images each.
case = str(sys.argv[1]) #'centered/'  # centered/ miscentered_0.1/ miscentered_peak/ #real/
training_or_test = str(sys.argv[2]) #'test' # training test validation
isolated_or_blended = str(sys.argv[3]) #'blended' # isolated blended
method_shift = str(sys.argv[4]) #'noshift' # 'noshift', 'uniform', 'uniform+betaprime'
do_peak_detection = str(sys.argv[5]).lower() == 'true' #False
N_files = int(sys.argv[6]) #1
nb_of_file_i = str(sys.argv[7])
N_per_file = 10000
assert training_or_test in ['training', 'validation', 'test']

# Method to shift centered galaxy
if isolated_or_blended == 'isolated':
    # where to save images and data
    save_dir = '/sps/lsst/users/barcelin/data/isolated_galaxies/' + case + training_or_test
    # what to call those files
    root = 'galaxies_isolated_20191024_'
    nmax_blend = 1
elif isolated_or_blended == 'blended':
    # where to save images and data
    save_dir = '/sps/lsst/users/barcelin/data/blended_galaxies/' + case + training_or_test
    # what to call those files
    root = 'galaxies_blended_20191024_'
    nmax_blend = 4
else:
    raise NotImplementedError
# Loading the COSMOS catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample') #dir=os.path.join(galsim.meta_data.share_dir,'COSMOS_25.2_training_sample'))#
cosmos_cat_dir = '/sps/lsst/users/barcelin/COSMOS_25.2_training_sample'
# Select galaxies to keep for the test sample
if training_or_test == 'test':
    used_idx = np.arange(5000)
else:
    used_idx = np.arange(5000,cosmos_cat.nobjects)

# keys for data objects
keys = ['nb_blended_gal', 'SNR', 'SNR_peak', 'redshift', 'moment_sigma', 'e1', 'e2', 'mag', 'mag_ir', 'closest_x', 'closest_y', 'closest_redshift', 'closest_moment_sigma', 'closest_e1', 'closest_e2', 'closest_mag', 'closest_mag_ir', 'blendedness_total_lsst', 'blendedness_closest_lsst', 'blendedness_aperture_lsst', 'idx_closest_to_peak', 'n_peak_detected']

for icat in trange(N_files):
    # Run params
    root_i = root+nb_of_file_i#str(icat)

    galaxies = []
    shifts = []
    if training_or_test == 'test':
        df = pd.DataFrame(index=np.arange(N_per_file), columns=keys)

    res = utils.apply_ntimes(image_generator_real, N_per_file, (cosmos_cat_dir, training_or_test, isolated_or_blended, save_dir, used_idx, nmax_blend, 100, 27.5, method_shift, do_peak_detection))
    for i in trange(N_per_file):
        if training_or_test == 'test':
            gal_noiseless, blend_noisy, data, shift = res[i]
            assert set(data.keys()) == set(keys)
            df.loc[i] = [data[k] for k in keys]
            shifts.append(shift)
        else:
            gal_noiseless, blend_noisy = res[i]
        galaxies.append((gal_noiseless, blend_noisy))

    # Save noisy blended images and denoised single central galaxy images
    np.save(os.path.join(save_dir, root_i+'_images.npy'), galaxies)

    # If the created sample is a test sample, also save the shifts and differents data
    if training_or_test == 'test':
        # Compute the normalizing constants for the generated sample
        noisy = np.array(galaxies)[:,1]
        max_i_noisy = []
        for i in range (10):
            max_i_noisy.append(np.max(noisy[:,i], axis = (1,2)))
        np.save(os.path.join(save_dir, root_i+'_I_norm.npy'), np.array(np.mean(max_i_noisy, axis = 1)))

        df.to_csv(os.path.join(save_dir, root_i+'_data.csv'), index=False)
        np.save(os.path.join(save_dir, root_i+'_shifts.npy'), np.array(shifts))
    
        del df
    del galaxies, res, shifts