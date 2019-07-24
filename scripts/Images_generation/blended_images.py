
# Import packages

import numpy as np
import matplotlib.pyplot as plt
import keras
import sys
import os
import logging
import galsim
import cmath as cm
import math
import random
import scipy
from scipy.stats import norm
from astropy.io import fits
from astropy.cosmology import WMAP9 as cosmo

sys.path.insert(0,'../tools_for_VAE/')
from tools_for_VAE import utils


# Import catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample')


path, filename = os.path.split('__file__')    
datapath = galsim.meta_data.share_dir
datapath2 = os.path.abspath(os.path.join(path,'/sps/lsst/users/barcelin/EUCLID_Filters/'))

# initialize (pseudo-)random number generator
random_seed = 1234567
rng = galsim.BaseDeviate(random_seed+1)

        # read in the Euclid NIR filters
filter_names_euclid_nir = 'HJY'
filter_names_euclid_vis = 'V'

# read in the LSST filters
filter_names_lsst = 'ugrizy'
filters = {}


for filter_name in filter_names_euclid_nir:
    filter_filename = os.path.join(datapath2, 'Euclid_NISP0.{0}.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='Angstrom')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

for filter_name in filter_names_euclid_vis:
    filter_filename = os.path.join(datapath2, 'Euclid_VIS.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='Angstrom')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

for filter_name in filter_names_lsst:
    filter_filename = os.path.join(datapath, 'bandpasses/LSST_{0}.dat'.format(filter_name))
    filters[filter_name] = galsim.Bandpass(filter_filename, wave_type='nm')
    filters[filter_name] = filters[filter_name].thin(rel_err=1e-4)

pixel_scale_lsst = 0.2 # arcseconds # LSST Science book
pixel_scale_euclid_nir = 0.3 # arcseconds # Euclid Science book
pixel_scale_euclid_vis = 0.1 # arcseconds # Euclid Science book
pixel_scale = [pixel_scale_euclid_nir]*3 + [pixel_scale_euclid_vis] + [pixel_scale_lsst]*6

#################### NOISE ###################
# Poissonian noise according to sky_level
N_exposures_lsst = 100
N_exposures_euclid = 1

sky_level_lsst_u = (2.512 **(26.50-22.95)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_g = (2.512 **(28.30-22.24)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_r = (2.512 **(28.13-21.20)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_i = (2.512 **(27.79-20.47)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_z = (2.512 **(27.40-19.60)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_lsst_y = (2.512 **(26.58-18.63)) * N_exposures_lsst # in e-.s-1.arcsec_2
sky_level_pixel_lsst = [int((sky_level_lsst_u * 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_g* 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_r* 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_i* 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_z* 15 * pixel_scale_lsst**2)),
                        int((sky_level_lsst_y* 15 * pixel_scale_lsst**2))]# in e-/pixel/15s

# average background level for Euclid observations : 22.35 mAB.arcsec-2 in VIS (Consortium book) ##
# For NIR bands, a coefficient is applied : it is calculated by comparing magnitudes AB of one point in the
# sky to the magnitude AB in VIS on this point. The choosen point is (-30;30) in galactic coordinates 
# (EUCLID and LSST overlap on this point).
coeff_noise_y = (22.57/21.95)
coeff_noise_j = (22.53/21.95)
coeff_noise_h = (21.90/21.95)

sky_level_nir_Y = (2.512 **(24.25-22.35*coeff_noise_y)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_nir_J = (2.512 **(24.29-22.35*coeff_noise_j)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_nir_H = (2.512 **(24.92-22.35*coeff_noise_h)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_vis = (2.512 **(25.58-22.35)) * N_exposures_euclid # in e-.s-1.arcsec_2
sky_level_pixel_nir = [int((sky_level_nir_Y * 1800 * pixel_scale_euclid_nir**2)),
                        int((sky_level_nir_J* 1800 * pixel_scale_euclid_nir**2)),
                        int((sky_level_nir_H* 1800 * pixel_scale_euclid_nir**2))]# in e-/pixel/1800s
sky_level_pixel_vis = int((sky_level_vis * 1800 * pixel_scale_euclid_vis**2))# in e-/pixel/1800s

sky_level = sky_level_pixel_nir + [sky_level_pixel_vis] + sky_level_pixel_lsst

########### PSF #####################
fwhm_lsst = 0.65 #pdf.rvs() #Fixed at median value : Fig 1 : https://arxiv.org/pdf/0805.2366.pdf

fwhm_euclid_nir = 0.22 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
fwhm_euclid_vis = 0.18 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
beta = 2.5
PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)#galsim.Moffat(fwhm=fwhm_lsst, beta=beta)
PSF_euclid_nir = galsim.Moffat(fwhm=fwhm_euclid_nir, beta=beta)
PSF_euclid_vis = galsim.Moffat(fwhm=fwhm_euclid_vis, beta=beta)


############# SIZE OF STAMPS ################
# The stamp size of NIR instrument is taken equal to the one of LSST to have a nb of pixels which is 
# an integer and in the same time the max_stamp_size a power of 2 (works better for FFT)
# The physical size in NIR instrument is then larger than in the other stamps but the information needed
# for the VAE and deblender is contained in the stamp.
phys_stamp_size = 6.4 # Arcsecond
lsst_stamp_size = int(phys_stamp_size/pixel_scale_lsst) # Nb of pixels
nir_stamp_size = int(phys_stamp_size/pixel_scale_euclid_nir)+1 # Nb of pixels
vis_stamp_size = int(phys_stamp_size/pixel_scale_euclid_vis) # Nb of pixels

max_stamp_size = np.max((lsst_stamp_size,nir_stamp_size,vis_stamp_size))



# Coefficients to take the exposure time and the telescope size into account
coeff_hst_lsst = (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * N_exposures_lsst
coeff_hst_euclid = (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * N_exposures_euclid


def get_scale_radius(gal):
    """
    Return the scale radius of the created galaxy
    
    Parameter:
    ---------
    gal: galaxy from which the scale radius is needed
    """
    try:
        return gal.obj_list[1].original.scale_radius
    except:
        return gal.original.scale_radius


#shift_method='uniform'
shift_method='lognorm_rad'

def shift_gal(gal, method='uniform'):
    """
    Return galaxy shifted according to the chosen shifting method
    
    Parameters:
    ----------
    gal: galaxy to shift (GalSim object)
    method: method to use for shifting
    """
    if method == 'uniform':
        shift_x = np.random.uniform(-2.5,2.5)  #(-1,1)
        shift_y = np.random.uniform(-2.5,2.5)  #(-1,1)
    elif method == 'lognorm_rad':
        scale_radius = get_scale_radius(gal)
        sample_x = np.random.lognormal(mean=1*scale_radius,sigma=1*scale_radius,size=None)
        shift_x = np.random.choice((sample_x, -sample_x), 1)[0]
        sample_y = np.random.lognormal(mean=1*scale_radius,sigma=1*scale_radius,size=None)
        shift_y = np.random.choice((sample_y, -sample_y), 1)[0]
    else:
        raise ValueError
    return gal.shift((shift_x,shift_y)), (shift_x,shift_y)




def create_images(i,filter_, sky_level_pixel, stamp_size, pixel_scale, nb_blended_gal, PSF, bdgal, add_gal):
    """
    Return images of the noiseless and noisy single galaxy and blend of galaxies
    
    Parameters:
    ----------
    i: number of bandpass filter (from 0 to 9: [0,1,2] are NIR, [3] is VIS, [4,5...,9] are LSST)
    filter_: filter to use to create the image
    sky_level_pixel: sky background
    stamp_size: size of the image
    pixel_scale: pixel scale corresponding to the used instrument
    nb_blended_gal: number of galaxies to add on the blended images
    PSF: PSF to use to create the image
    bdgal: galaxy to add on the center of the image
    add_gal: galaxies to add on the blended images
    """
    # Create poissonian noise
    poissonian_noise = galsim.PoissonNoise(rng, sky_level=sky_level_pixel)
    
    # Create images to be filled with galaxies and noise
    img = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)
    img_noisy = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)
    img_blend_noiseless = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)
    img_blend_noisy = galsim.ImageF(stamp_size,stamp_size, scale=pixel_scale)

    # Create noiseless image of the centered galaxy
    bdfinal = galsim.Convolve([bdgal, PSF])

    # Initialize each image: mandatory or allocation problems
    bdfinal.drawImage(filter_, image=img)
    bdfinal.drawImage(filter_, image=img_noisy)
    bdfinal.drawImage(filter_, image=img_blend_noiseless)
    bdfinal.drawImage(filter_, image=img_blend_noisy)

    # Initialize blendedness lists
    Blendedness_lsst = np.zeros((nb_blended_gal-1))
    Blendedness_euclid = np.zeros((nb_blended_gal-1))

    # Save noiseless galaxy
    galaxy_noiseless = img.array.data
    #img_noiseless = img

    # Add noise and save noisy centered galaxy
    img_noisy.addNoise(poissonian_noise)
    galaxy_noisy = img_noisy.array.data

    # Choose value to create blended image in NIR, VIS or LSST filter
    if i < 3:
        rank_img = 0
    elif i == 3:
        rank_img = 1
    else:
        rank_img = 2

    # Create blended image
    for k in range (nb_blended_gal-1):
        img_new = galsim.ImageF(stamp_size, stamp_size, scale=pixel_scale)
        bdfinal_new = galsim.Convolve([add_gal[k][rank_img], PSF])
        bdfinal_new.drawImage(filter_, image=img_new)
        img_blend_noiseless += img_new
        img_blend_noisy += img_new
        if i == 3 :
            Blendedness_euclid[k]= utils.blendedness(img,img_new)
        elif i ==6 :
            Blendedness_lsst[k]= utils.blendedness(img,img_new)
    
    # Save noiseless blended image
    blend_noiseless = img_blend_noiseless.array.data
    
    # Add noise and save noisy blended image
    img_blend_noisy.addNoise(poissonian_noise)
    blend_noisy = img_blend_noisy.array.data

    return galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy, Blendedness_lsst, Blendedness_euclid



# Generation function
def Gal_generator_noisy_test_2(cosmos_cat, nb_blended_gal, training_or_test):
    """
    Return numpy arrays: noiseless and noisy image of single galaxy and of blended galaxies

    Parameters:
    ----------
    cosmos_cat: COSMOS catalog
    nb_blended_gal: number of galaxies to add to the centered one on the blended image
    training_or_test: choice for generating a training or testing dataset
    """
    ############## GENERATION OF THE GALAXIES ##################
    ud = galsim.UniformDeviate()
    
    galaxies = []
    mag=[]
    for i in range (nb_blended_gal):
        galaxies.append(cosmos_cat.makeGalaxy(random.randint(0,cosmos_cat.nobjects-1), gal_type='parametric', chromatic=True, noise_pad_size = 0))
        mag.append(galaxies[i].calculateMagnitude(filters['r'].withZeropoint(28.13)))

    gal = galaxies[np.where(mag == np.min(mag))[0][0]]
    redshift = gal.SED.redshift
    
    galaxies.remove(gal)
        
    ############ LUMINOSITY ############# 
    # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
    bdgal_lsst =  gal * coeff_hst_lsst
    bdgal_euclid_nir =  coeff_hst_euclid * gal
    bdgal_euclid_vis =  coeff_hst_euclid * gal

    add_gal = []
    shift=np.zeros((nb_blended_gal-1, 2))
    for i in range (len(galaxies)):
        gal_new = None
        ud = galsim.UniformDeviate()
        gal_new = galaxies[i].rotate(ud() * 360. * galsim.degrees)

        gal_new, shift[i]  = shift_gal(gal_new, method=shift_method)
            
        bdgal_new_lsst = None
        bdgal_new_euclid_nir =None
        bdgal_new_euclid_vis =None
        bdgal_new_lsst =  coeff_hst_lsst * gal_new
        bdgal_new_euclid_nir =  coeff_hst_euclid * gal_new
        bdgal_new_euclid_vis =  coeff_hst_euclid * gal_new
        add_gal.append([bdgal_new_euclid_nir,bdgal_new_euclid_vis, bdgal_new_lsst])

    galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
    galaxy_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

    blend_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
    blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

    Blendedness_lsst = np.zeros((10,nb_blended_gal-1))
    Blendedness_euclid = np.zeros((10,nb_blended_gal-1))

    i = 0
    for filter_name, filter_ in filters.items():
        if (i < 3):
            galaxy_noiseless[i], galaxy_noisy[i], blend_noiseless[i], blend_noisy[i], Blendedness_lsst[i], Blendedness_euclid[i] = create_images(i, filter_,sky_level_pixel_nir[i], max_stamp_size, pixel_scale_euclid_nir, nb_blended_gal, PSF_euclid_nir, bdgal_euclid_nir, add_gal)
        elif (i==3):
            galaxy_noiseless[i], galaxy_noisy[i], blend_noiseless[i], blend_noisy[i], Blendedness_lsst[i], Blendedness_euclid[i] = create_images(i, filter_,sky_level_pixel_vis, max_stamp_size, pixel_scale_euclid_vis, nb_blended_gal, PSF_euclid_vis, bdgal_euclid_vis, add_gal)
        else:
            galaxy_noiseless[i], galaxy_noisy[i], blend_noiseless[i], blend_noisy[i], Blendedness_lsst[i], Blendedness_euclid[i] = create_images(i, filter_,sky_level_pixel_lsst[i-4], max_stamp_size, pixel_scale_lsst, nb_blended_gal, PSF_lsst, bdgal_lsst, add_gal)
        i+=1
    
    # Return outputs depending on the kind of generated dataset
    if training_or_test == 'training':
        return galaxy_noiseless, blend_noisy
    if training_or_test == 'test':
        return galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy, shift, mag, Blendedness_euclid[3], Blendedness_lsst[6]




























# Generation function
def Gal_generator_noisy_test(cosmos_cat, nb_blended_gal):
    count = 0
    galaxy = np.zeros((10))
    while (galaxy.all() == 0):
        try:
            ############## GENERATION OF THE GALAXIES ##################
            ud = galsim.UniformDeviate()
            
            galaxies = []
            mag=[]
            for i in range (nb_blended_gal):
                galaxies.append(cosmos_cat.makeGalaxy(random.randint(0,cosmos_cat.nobjects-1), gal_type='parametric', chromatic=True, noise_pad_size = 0))
                mag.append(galaxies[i].calculateMagnitude(filters['r'].withZeropoint(28.13)))

            gal = galaxies[np.where(mag == np.min(mag))[0][0]]
            redshift = gal.SED.redshift
            
            galaxies.remove(gal)
                
            ############ LUMINOSITY ############# 
            # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
            bdgal_lsst =  gal * coeff_hst_lsst
            bdgal_euclid_nir =  coeff_hst_euclid * gal
            bdgal_euclid_vis =  coeff_hst_euclid * gal


            add_gal = []
            shift=np.zeros((nb_blended_gal-1, 2))
            for i in range (len(galaxies)):
                gal_new = None
                ud = galsim.UniformDeviate()
                gal_new = galaxies[i].rotate(ud() * 360. * galsim.degrees)

                gal_new, shift[i]  = shift_gal(gal_new, method=shift_method)
                   
                bdgal_new_lsst = None
                bdgal_new_euclid_nir =None
                bdgal_new_euclid_vis =None
                bdgal_new_lsst =  coeff_hst_lsst * gal_new
                bdgal_new_euclid_nir =  coeff_hst_euclid * gal_new
                bdgal_new_euclid_vis =  coeff_hst_euclid * gal_new
                add_gal.append([bdgal_new_euclid_nir,bdgal_new_euclid_vis, bdgal_new_lsst])

            #print(len(add_gal),len(add_gal[0]))
            ########### PSF #####################
            # convolve with PSF to make final profil : profil from LSST science book and (https://arxiv.org/pdf/0805.2366.pdf)
            # mu = -0.43058681997903414
            # sigma = 0.3404334041976153
            # p_unnormed = lambda x : (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
            #                    / (x * sigma * np.sqrt(2 * np.pi)))#((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
            # p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
            # p = lambda z : p_unnormed(z) / p_normalization

            # from scipy import stats
            # class PSF_distribution(stats.rv_continuous):
            #     def __init__(self):
            #         super(PSF_distribution, self).__init__()
            #         self.a = 0.
            #         self.b = 10.
            #     def _pdf(self, x):
            #         return p(x)

            # pdf = PSF_distribution()
            # fwhm_lsst = 0.65 #pdf.rvs() #Fixed at median value : Fig 1 : https://arxiv.org/pdf/0805.2366.pdf

            # fwhm_euclid_nir = 0.22 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
            # fwhm_euclid_vis = 0.18 # EUCLID PSF is supposed invariant (no atmosphere) despite the optical and wavelengths variations
            # beta = 2.5
            # PSF_lsst = galsim.Kolmogorov(fwhm=fwhm_lsst)#galsim.Moffat(fwhm=fwhm_lsst, beta=beta)
            # PSF_euclid_nir = galsim.Moffat(fwhm=fwhm_euclid_nir, beta=beta)
            # PSF_euclid_vis = galsim.Moffat(fwhm=fwhm_euclid_vis, beta=beta)

            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            galaxy_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

            blend_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

            Blendedness_lsst = np.zeros((nb_blended_gal-1))
            Blendedness_euclid = np.zeros((nb_blended_gal-1))


            #image = np.zeros((64,64))
            #image_new = np.zeros ((64,64))

            i = 0
            for filter_name, filter_ in filters.items():
                #print('i = '+str(i))
                if (i < 3):
                    #print('in NIR')
                    poissonian_noise_nir = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_nir[i])
                    img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)  
                    bdfinal = galsim.Convolve([bdgal_euclid_nir, PSF_euclid_nir])
                    bdfinal.drawImage(filter_, image=img)
                    # Noiseless galaxy
                    galaxy_noiseless[i] = img.array.data
                    img_noiseless = img

                    #### Bended image
                    img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)
                    img_blended = img_noiseless
                    for k in range (nb_blended_gal-1):
                        img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_nir)
                        bdfinal_new = galsim.Convolve([add_gal[k][0], PSF_euclid_nir])
                        bdfinal_new.drawImage(filter_, image=img_new)
                        img_blended += img_new
                    # Noiseless blended image
                    blend_noiseless[i] = img_blended.array.data


                    # Noisy centered galaxy
                    img_blended.addNoise(poissonian_noise_nir)
                    blend_noisy[i] = img_blended.array.data

                    # Noisy galaxy
                    img.addNoise(poissonian_noise_nir)
                    galaxy_noisy[i] = img.array.data
                else:
                    if (i==3):
                        poissonian_noise_vis = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_vis)
                        img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)  
                        bdfinal = galsim.Convolve([bdgal_euclid_vis, PSF_euclid_vis])
                        bdfinal.drawImage(filter_, image=img)
                        # Noiseless galaxy
                        galaxy_noiseless[i] = img.array.data
                        img_noiseless = img

                        #### Bended image
                        img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)
                        img_blended = img_noiseless
                        for k in range (nb_blended_gal-1):
                            img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_vis)
                            bdfinal_new = galsim.Convolve([add_gal[k][1], PSF_euclid_vis])
                            bdfinal_new.drawImage(filter_, image=img_new)
                            img_blended += img_new
                            Blendedness_euclid[k]= utils.blendedness(img,img_new)
                        # Noiseless blended image 
                        blend_noiseless[i] = img_blended.array.data
                        # Noisy centered galaxy
                        img_blended.addNoise(poissonian_noise_vis)
                        blend_noisy[i] = img_blended.array.data

                        # Noisy galaxy
                        img.addNoise(poissonian_noise_vis)
                        galaxy_noisy[i] = img.array.data

                    else:
        #                print('passage a LSST')
                        poissonian_noise_lsst = galsim.PoissonNoise(rng, sky_level_pixel_lsst[i-4])
                        img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)  
                        bdfinal = galsim.Convolve([bdgal_lsst, PSF_lsst])
                        bdfinal.drawImage(filter_, image=img)
                        #bdfinal.drawImage(filter_, image=img_noiseless)
                        # Noiseless galaxy
                        galaxy_noiseless[i] = img.array.data
                        img_noiseless = img

                        # Noisy galaxy
                        img.addNoise(poissonian_noise_lsst)
                        galaxy_noisy[i]= img.array.data

                        #### Bended image
                        img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)
                        img_blended = img_noiseless
                        for k in range (nb_blended_gal-1):
                            img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_lsst)
                            bdfinal_new = galsim.Convolve([add_gal[k][2], PSF_lsst])
                            bdfinal_new.drawImage(filter_, image=img_new)
                            img_blended += img_new
                            if (i==6):
                                Blendedness_lsst[k]= utils.blendedness(img,img_new)
                        # Noiseless blended image
                        blend_noiseless[i] = img_blended.array.data
                        # Noisy centered galaxy
                        img_blended.addNoise(poissonian_noise_lsst)
                        blend_noisy[i] = img_blended.array.data




                i+=1
            return galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy, shift, mag, Blendedness_euclid, Blendedness_lsst
        except RuntimeError: 
            count +=1
    print("nb of error : "+(count))
    
    
# Generation function
def Gal_generator_noisy_training(cosmos_cat, nb_blended_gal):
    cosmos_cat = cosmos_cat
    count = 0
    galaxy = np.zeros((10))
    while (galaxy.all() == 0):
        try:
            ############## GENERATION OF THE GALAXIES ##################
            ud = galsim.UniformDeviate()
            
            galaxies = []
            mag=[]
            for i in range (nb_blended_gal):
                galaxies.append(cosmos_cat.makeGalaxy(random.randint(0,cosmos_cat.nobjects-1), gal_type='parametric', chromatic=True, noise_pad_size = 0))
                mag.append(galaxies[i].calculateMagnitude(filters['r'].withZeropoint(28.13)))

            gal = galaxies[np.where(mag == np.min(mag))[0][0]]
            redshift = gal.SED.redshift
            
            galaxies.remove(gal)
                
            ############ LUMINOSITY ############# 
            # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
            bdgal_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_lsst
            bdgal_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_euclid
            bdgal_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures_euclid         


            add_gal = []
            shift=np.zeros((nb_blended_gal-1, 2))
            for i in range (len(galaxies)):
                gal_new = None
                ud = galsim.UniformDeviate()
                gal_new = galaxies[i].rotate(ud() * 360. * galsim.degrees)
                shift_x = np.random.uniform(-2.5,2.5)
                shift_y = np.random.uniform(-2.5,2.5)

                gal_new = gal_new.shift((shift_x,shift_y))
                   
                bdgal_new_lsst = None
                bdgal_new_euclid_nir =None
                bdgal_new_euclid_vis =None
                bdgal_new_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures_lsst
                bdgal_new_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures_euclid
                bdgal_new_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures_euclid
                add_gal.append([bdgal_new_euclid_nir,bdgal_new_euclid_vis, bdgal_new_lsst])
                shift[i]=(shift_x,shift_y)

            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            blend_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

            i = 0
            for filter_name, filter_ in filters.items():
                #print('i = '+str(i))
                if (i < 3):
                    #print('in NIR')
                    poissonian_noise_nir = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_nir[i])
                    img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)  
                    bdfinal = galsim.Convolve([bdgal_euclid_nir, PSF_euclid_nir])
                    bdfinal.drawImage(filter_, image=img)
                    # Noiseless galaxy
                    galaxy_noiseless[i] = img.array.data
                    img_noiseless = img
                    #print('single galaxy finished')

                    #### Bended image
                    img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)
                    img_blended = img_noiseless
                    for k in range (nb_blended_gal-1):
                        img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_nir)
                        bdfinal_new = galsim.Convolve([add_gal[k][0], PSF_euclid_nir])
                        bdfinal_new.drawImage(filter_, image=img_new)
                        img_blended = img_blended + img_new
                    # Noisy centered galaxy
                    img_blended.addNoise(poissonian_noise_nir)
                    blend_noisy[i] = img_blended.array.data

                else:
                    if (i==3):
                        poissonian_noise_vis = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_vis)
                        img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)  
                        bdfinal = galsim.Convolve([bdgal_euclid_vis, PSF_euclid_vis])
                        bdfinal.drawImage(filter_, image=img)
                        # Noiseless galaxy
                        galaxy_noiseless[i] = img.array.data
                        img_noiseless = img

                        #### Bended image
                        img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)
                        img_blended = img_noiseless
                        for k in range (nb_blended_gal-1):
                            img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_vis)
                            bdfinal_new = galsim.Convolve([add_gal[k][1], PSF_euclid_vis])
                            bdfinal_new.drawImage(filter_, image=img_new)
                            img_blended = img_blended + img_new
                        # Noisy centered galaxy
                        img_blended.addNoise(poissonian_noise_vis)
                        blend_noisy[i] = img_blended.array.data


                    else:
        #                print('passage a LSST')
                        poissonian_noise_lsst = galsim.PoissonNoise(rng, sky_level_pixel_lsst[i-4])
                        img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)  
                        bdfinal = galsim.Convolve([bdgal_lsst, PSF_lsst])
                        bdfinal.drawImage(filter_, image=img)
                        bdfinal.drawImage(filter_, image=img_noiseless)
                        # Noiseless galaxy
                        galaxy_noiseless[i] = img.array.data
                        img_noiseless = img

                        #### Bended image
                        img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)
                        img_blended = img_noiseless
                        for k in range (nb_blended_gal-1):
                            img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_lsst)
                            bdfinal_new = galsim.Convolve([add_gal[k][2], PSF_lsst])
                            bdfinal_new.drawImage(filter_, image=img_new)
                            img_blended = img_blended + img_new
                        # Noisy centered galaxy
                        img_blended.addNoise(poissonian_noise_lsst)
                        blend_noisy[i] = img_blended.array.data

                i+=1
            return galaxy_noiseless, blend_noisy
        except RuntimeError: 
            count +=1
    print("nb of error : "+(count))
 
    