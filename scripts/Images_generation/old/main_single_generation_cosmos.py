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
from tqdm.auto import tqdm, trange

from cosmos_generation import cosmos_galaxy_generator

sys.path.insert(0,'../tools_for_VAE/')
import tools_for_VAE
from tools_for_VAE import utils

# The script is used as, eg,
# # python main_single_generation_cosmos.py training 10 1000
# to produce 10 files in the training sample with 1000 images each.
training_or_test = 'training' #str(sys.argv[1])
N_files = 1 #int(sys.argv[2])
N_per_file = 10000 #int(sys.argv[3])
assert training_or_test in ['training', 'validation', 'test']
#training_or_test = 'training'

# where to save images and data
save_dir = '/sps/lsst/users/barcelin/data/' + training_or_test #single_galaxies/
# what to call those files
root = 'galaxies_isolated_20191022_'

# Loading the COSMOS catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample')
cosmos_cat_filename = '/sps/lsst/users/barcelin/COSMOS_25.2_training_sample/real_galaxy_catalog_25.2.fits'
# Select galaxies to keep for the test sample
if training_or_test == 'test':
    used_idx = np.arange(5000)
else:
    used_idx = np.arange(5000,cosmos_cat.nobjects)

for icat in trange(N_files):
    # Run params
    root_i = root+str(icat)

    galaxies = []
    df = pd.DataFrame(index=np.arange(N_per_file), columns=['redshift', 'moment_sigma', 'e1', 'e2', 'mag', 'SNR', 'SNR_peak'])

    res = utils.apply_ntimes(cosmos_galaxy_generator, N_per_file, (cosmos_cat_filename, training_or_test, used_idx, 100, 28))

    for i in trange(N_per_file):
        if training_or_test == 'test':
            gal_noiseless, gal_noisy, data = res[i]
            df.loc[i] = np.array(data)
        else:
            gal_noiseless, gal_noisy = res[i]
        galaxies.append((gal_noiseless,gal_noisy))

    # Save noisy blended images and denoised single central galaxy images
    np.save('/sps/lsst/users/barcelin/data/single_galaxies/28/training/galaxies_isolated_20191022_16_images.npy', galaxies)#os.path.join(save_dir, root_i+       single_galaxies/training
    # If the created sample is a test sample, also save the shifts and differents data
    if training_or_test == 'test':
        df.to_csv(os.path.join(save_dir, root_i+'_data.csv'), index=False)
