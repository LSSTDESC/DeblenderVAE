
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

# Import catalog
cosmos_cat = galsim.COSMOSCatalog('real_galaxy_catalog_25.2.fits', dir='/sps/lsst/users/barcelin/COSMOS_25.2_training_sample')

# Generation function
def Gal_generator_noisy_test(cosmos_cat, nb_blended_gal):
    cosmos_cat = cosmos_cat
    count = 0
    galaxy = np.zeros((10))
    while (galaxy.all() == 0):
        try:
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


            #################### NOISE ###################
            # Poissonian noise according to sky_level
            N_exposures = 100


            sky_level_lsst_u = (2.512 **(26.50-22.95)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_g = (2.512 **(28.30-22.24)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_r = (2.512 **(28.13-21.20)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_i = (2.512 **(27.79-20.47)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_z = (2.512 **(27.40-19.60)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_y = (2.512 **(26.58-18.63)) * N_exposures # in e-.s-1.arcsec_2
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

            sky_level_nir_Y = (2.512 **(24.25-22.35*coeff_noise_y)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_nir_J = (2.512 **(24.29-22.35*coeff_noise_j)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_nir_H = (2.512 **(24.92-22.35*coeff_noise_h)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_vis = (2.512 **(25.58-22.35)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_pixel_nir = [int((sky_level_nir_Y * 1800 * pixel_scale_euclid_nir**2)),
                                   int((sky_level_nir_J* 1800 * pixel_scale_euclid_nir**2)),
                                   int((sky_level_nir_H* 1800 * pixel_scale_euclid_nir**2))]# in e-/pixel/1800s
            sky_level_pixel_vis = int((sky_level_vis * 1800 * pixel_scale_euclid_vis**2))# in e-/pixel/1800s


            # 25.94 : zeros point for the makeGalaxy method and normalization: http://galsim-developers.github.io/GalSim/classgalsim_1_1scene_1_1_c_o_s_m_o_s_catalog.html            
            sky_level_cosmos = 10**((25.94-22.35)/2.5)            




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
            bdgal_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures
            bdgal_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures
            bdgal_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures         


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
                bdgal_new_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures
                bdgal_new_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures
                bdgal_new_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures
                add_gal.append([bdgal_new_euclid_nir,bdgal_new_euclid_vis, bdgal_new_lsst])
                shift[i]=(shift_x,shift_y)
            
            #print(len(add_gal),len(add_gal[0]))
            ########### PSF #####################
            # convolve with PSF to make final profil : profil from LSST science book and (https://arxiv.org/pdf/0805.2366.pdf)
            mu = -0.43058681997903414
            sigma = 0.3404334041976153
            p_unnormed = lambda x : (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
                               / (x * sigma * np.sqrt(2 * np.pi)))#((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
            p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
            p = lambda z : p_unnormed(z) / p_normalization

            from scipy import stats
            class PSF_distribution(stats.rv_continuous):
                def __init__(self):
                    super(PSF_distribution, self).__init__()
                    self.a = 0.
                    self.b = 10.
                def _pdf(self, x):
                    return p(x)

            pdf = PSF_distribution()
            fwhm_lsst = pdf.rvs()

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

            galaxy_nir_noiseless = np.zeros((3,max_stamp_size,max_stamp_size))
            galaxy_vis_noiseless = np.zeros((1,max_stamp_size,max_stamp_size))
            galaxy_lsst_noiseless = np.zeros((6,max_stamp_size,max_stamp_size))
            galaxy_nir_noisy = np.zeros((3,max_stamp_size,max_stamp_size))
            galaxy_vis_noisy = np.zeros((1,max_stamp_size,max_stamp_size))
            galaxy_lsst_noisy = np.zeros((6,max_stamp_size,max_stamp_size))
            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            galaxy_noisy = np.zeros((10,max_stamp_size,max_stamp_size))


            blend_nir_noiseless = np.zeros((3,max_stamp_size,max_stamp_size))
            blend_vis_noiseless = np.zeros((1,max_stamp_size,max_stamp_size))
            blend_lsst_noiseless = np.zeros((6,max_stamp_size,max_stamp_size))
            blend_nir_noisy = np.zeros((3,max_stamp_size,max_stamp_size))
            blend_vis_noisy = np.zeros((1,max_stamp_size,max_stamp_size))
            blend_lsst_noisy = np.zeros((6,max_stamp_size,max_stamp_size))
            blend_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
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
                    galaxy_nir_noiseless[i]= img.array.data
                    galaxy_noiseless[i] = galaxy_nir_noiseless[i]
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
                    # Noiseless blended image
                    blend_nir_noiseless[i]= img_blended.array.data
                    blend_noiseless[i] = blend_nir_noiseless[i]
                    # Noisy centered galaxy
                    img_blended.addNoise(poissonian_noise_nir)
                    blend_nir_noisy[i]= img_blended.array.data
                    blend_noisy[i] = blend_nir_noisy[i]

                    # Noisy galaxy
                    img.addNoise(poissonian_noise_nir)
                    galaxy_nir_noisy[i]= img.array.data
                    galaxy_noisy[i] = galaxy_nir_noisy[i]
                else:
                    if (i==3):
                        poissonian_noise_vis = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_vis)
                        img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)  
                        bdfinal = galsim.Convolve([bdgal_euclid_vis, PSF_euclid_vis])
                        bdfinal.drawImage(filter_, image=img)
                        # Noiseless galaxy
                        galaxy_vis_noiseless[3-i]= img.array.data
                        galaxy_noiseless[i] = galaxy_vis_noiseless[3-i]
                        img_noiseless = img

                        #### Bended image
                        img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)
                        img_blended = img_noiseless
                        for k in range (nb_blended_gal-1):
                            img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_euclid_vis)
                            bdfinal_new = galsim.Convolve([add_gal[k][1], PSF_euclid_vis])
                            bdfinal_new.drawImage(filter_, image=img_new)
                            img_blended = img_blended + img_new
                        # Noiseless blended image
                        blend_vis_noiseless[3-i]= img_blended.array.data
                        blend_noiseless[i] = blend_vis_noiseless[3-i]
                        # Noisy centered galaxy
                        img_blended.addNoise(poissonian_noise_vis)
                        blend_vis_noisy[3-i]= img_blended.array.data
                        blend_noisy[i] = blend_vis_noisy[3-i]

                        # Noisy galaxy
                        img.addNoise(poissonian_noise_vis)
                        galaxy_vis_noisy[3-i]= img.array.data
                        galaxy_noisy[i] = galaxy_vis_noisy[3-i]

                    else:
        #                print('passage a LSST')
                        poissonian_noise_lsst = galsim.PoissonNoise(rng, sky_level_pixel_lsst[i-4])
                        img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)  
                        bdfinal = galsim.Convolve([bdgal_lsst, PSF_lsst])
                        bdfinal.drawImage(filter_, image=img)
                        bdfinal.drawImage(filter_, image=img_noiseless)
                        # Noiseless galaxy
                        galaxy_lsst_noiseless[i-4]= img.array.data
                        galaxy_noiseless[i] = galaxy_lsst_noiseless[i-4]
                        img_noiseless = img

                        #### Bended image
                        img_blended = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)
                        img_blended = img_noiseless
                        for k in range (nb_blended_gal-1):
                            img_new = galsim.ImageF(max_stamp_size, max_stamp_size, scale=pixel_scale_lsst)
                            bdfinal_new = galsim.Convolve([add_gal[k][2], PSF_lsst])
                            bdfinal_new.drawImage(filter_, image=img_new)
                            img_blended = img_blended + img_new
                        # Noiseless blended image
                        blend_lsst_noiseless[i-4]= img_blended.array.data
                        blend_noiseless[i] = blend_lsst_noiseless[i-4]
                        # Noisy centered galaxy
                        img_blended.addNoise(poissonian_noise_lsst)
                        blend_lsst_noisy[i-4]= img_blended.array.data
                        blend_noisy[i] = blend_lsst_noisy[i-4]

                        # Noisy galaxy
                        img.addNoise(poissonian_noise_lsst)
                        galaxy_lsst_noisy[i-4]= img.array.data
                        galaxy_noisy[i]= galaxy_lsst_noisy[i-4]


                i+=1
            return galaxy_noiseless, galaxy_noisy, blend_noiseless, blend_noisy,shift, redshift
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


            #################### NOISE ###################
            # Poissonian noise according to sky_level
            N_exposures = 100


            sky_level_lsst_u = (2.512 **(26.50-22.95)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_g = (2.512 **(28.30-22.24)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_r = (2.512 **(28.13-21.20)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_i = (2.512 **(27.79-20.47)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_z = (2.512 **(27.40-19.60)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_lsst_y = (2.512 **(26.58-18.63)) * N_exposures # in e-.s-1.arcsec_2
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

            sky_level_nir_Y = (2.512 **(24.25-22.35*coeff_noise_y)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_nir_J = (2.512 **(24.29-22.35*coeff_noise_j)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_nir_H = (2.512 **(24.92-22.35*coeff_noise_h)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_vis = (2.512 **(25.58-22.35)) * N_exposures # in e-.s-1.arcsec_2
            sky_level_pixel_nir = [int((sky_level_nir_Y * 1800 * pixel_scale_euclid_nir**2)),
                                   int((sky_level_nir_J* 1800 * pixel_scale_euclid_nir**2)),
                                   int((sky_level_nir_H* 1800 * pixel_scale_euclid_nir**2))]# in e-/pixel/1800s
            sky_level_pixel_vis = int((sky_level_vis * 1800 * pixel_scale_euclid_vis**2))# in e-/pixel/1800s


            # 25.94 : zeros point for the makeGalaxy method and normalization: http://galsim-developers.github.io/GalSim/classgalsim_1_1scene_1_1_c_o_s_m_o_s_catalog.html            
            sky_level_cosmos = 10**((25.94-22.35)/2.5)            




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
            bdgal_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures
            bdgal_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures
            bdgal_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures         


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
                bdgal_new_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures
                bdgal_new_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures
                bdgal_new_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal_new * N_exposures
                add_gal.append([bdgal_new_euclid_nir,bdgal_new_euclid_vis, bdgal_new_lsst])
                shift[i]=(shift_x,shift_y)
            
            #print(len(add_gal),len(add_gal[0]))
            ########### PSF #####################
            # convolve with PSF to make final profil : profil from LSST science book and (https://arxiv.org/pdf/0805.2366.pdf)
            mu = -0.43058681997903414
            sigma = 0.3404334041976153
            p_unnormed = lambda x : (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
                               / (x * sigma * np.sqrt(2 * np.pi)))#((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
            p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
            p = lambda z : p_unnormed(z) / p_normalization

            from scipy import stats
            class PSF_distribution(stats.rv_continuous):
                def __init__(self):
                    super(PSF_distribution, self).__init__()
                    self.a = 0.
                    self.b = 10.
                def _pdf(self, x):
                    return p(x)

            pdf = PSF_distribution()
            fwhm_lsst = pdf.rvs()

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

            galaxy_nir_noiseless = np.zeros((3,max_stamp_size,max_stamp_size))
            galaxy_vis_noiseless = np.zeros((1,max_stamp_size,max_stamp_size))
            galaxy_lsst_noiseless = np.zeros((6,max_stamp_size,max_stamp_size))
            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))


            blend_nir_noisy = np.zeros((3,max_stamp_size,max_stamp_size))
            blend_vis_noisy = np.zeros((1,max_stamp_size,max_stamp_size))
            blend_lsst_noisy = np.zeros((6,max_stamp_size,max_stamp_size))
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
                    galaxy_nir_noiseless[i]= img.array.data
                    galaxy_noiseless[i] = galaxy_nir_noiseless[i]
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
                    blend_nir_noisy[i]= img_blended.array.data
                    blend_noisy[i] = blend_nir_noisy[i]

                else:
                    if (i==3):
                        poissonian_noise_vis = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_vis)
                        img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_vis)  
                        bdfinal = galsim.Convolve([bdgal_euclid_vis, PSF_euclid_vis])
                        bdfinal.drawImage(filter_, image=img)
                        # Noiseless galaxy
                        galaxy_vis_noiseless[3-i]= img.array.data
                        galaxy_noiseless[i] = galaxy_vis_noiseless[3-i]
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
                        blend_vis_noisy[3-i]= img_blended.array.data
                        blend_noisy[i] = blend_vis_noisy[3-i]


                    else:
        #                print('passage a LSST')
                        poissonian_noise_lsst = galsim.PoissonNoise(rng, sky_level_pixel_lsst[i-4])
                        img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_lsst)  
                        bdfinal = galsim.Convolve([bdgal_lsst, PSF_lsst])
                        bdfinal.drawImage(filter_, image=img)
                        bdfinal.drawImage(filter_, image=img_noiseless)
                        # Noiseless galaxy
                        galaxy_lsst_noiseless[i-4]= img.array.data
                        galaxy_noiseless[i] = galaxy_lsst_noiseless[i-4]
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
                        blend_lsst_noisy[i-4]= img_blended.array.data
                        blend_noisy[i] = blend_lsst_noisy[i-4]

                i+=1
            return galaxy_noiseless, blend_noisy
        except RuntimeError: 
            count +=1
    print("nb of error : "+(count))
 
    