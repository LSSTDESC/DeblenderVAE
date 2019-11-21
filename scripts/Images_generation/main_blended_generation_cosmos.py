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

from blended_images import blend_generator

sys.path.insert(0,'../tools_for_VAE/')
import tools_for_VAE
from tools_for_VAE import utils

from images_generator import image_generator, image_generator_2

# The script is used as, eg,
# # python main_blended_generation_cosmos.py training 10 1000
# to produce 10 files in the training sample with 1000 images each.
training_or_test = 'training'
isolated_or_blended = 'blended'
N_files = 1 
N_per_file = 10000
assert training_or_test in ['training', 'validation', 'test']

# where to save images and data
save_dir = '/sps/lsst/users/barcelin/data/blended_images/28/miscenter_19112019/' + training_or_test #blended_images#single_galaxies
# what to call those files
root = 'galaxies_blended_20191024_'

# Method to shift centered galaxy
if isolated_or_blended == 'isolated':
    method = 'uniform+betaprime'
    nmax_blend = 1
elif isolated_or_blended == 'blended':
    method = 'uniform'
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
keys = ['nb_blended_gal', 'SNR', 'SNR_peak', 'redshift', 'moment_sigma', 'e1', 'e2', 'mag', 'mag_ir', 'closest_x', 'closest_y', 'closest_redshift', 'closest_moment_sigma', 'closest_e1', 'closest_e2', 'closest_mag','closest_mag_ir', 'blendedness_total_lsst', 'blendedness_aperture_lsst', 'blendedness_closest_lsst']#'blendedness_total_euclid', , 'blendedness_closest_euclid'
for icat in trange(N_files):
    # Run params
    root_i = root+str(icat)

    galaxies = []
    shifts = []
    if training_or_test == 'test':
        df = pd.DataFrame(index=np.arange(N_per_file), columns=keys)

    #res = [image_generator_2(cosmos_cat_dir, training_or_test, isolated_or_blended, used_idx, nmax_blend, 100, 28.,method) for _ in range(N_per_file)]
    res = utils.apply_ntimes(image_generator_2, N_per_file, (cosmos_cat_dir, training_or_test, isolated_or_blended, used_idx, nmax_blend, 100, 28.,method))
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
    np.save(os.path.join(save_dir, root_i+'_4_images.npy'), galaxies)

    # If the created sample is a test sample, also save the shifts and differents data
    if training_or_test == 'test':
        df.to_csv(os.path.join(save_dir, root_i+'_data.csv'), index=False)
        np.save(os.path.join(save_dir, root_i+'_shifts.npy'), np.array(shifts))