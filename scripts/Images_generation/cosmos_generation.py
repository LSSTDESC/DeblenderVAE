
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
import random

# Generation function
def Gal_generator_noisy(cosmos_cat):
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

           # Chromatic bulge+disk galaxy
            # On prend la distribution de galaxies en fonction du redshift du Science Book de LSST
            i = 25.
            z0 = (0.0417*i)-0.744
            zl=np.arange(0.001,5,0.01)

            p_unnormed = lambda z : ((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
            p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
            p = lambda z : p_unnormed(z) / p_normalization

            from scipy import stats
            class LSST_redshift_distribution(stats.rv_continuous):
                def __init__(self):
                    super(LSST_redshift_distribution, self).__init__()
                    self.a = 0.
                    self.b = 10.
                def _pdf(self, x):
                    return p(x)

            nz = LSST_redshift_distribution()

            redshift = nz.rvs()   

            ############## SHAPE OF THE GALAXY ##################
            ud = galsim.UniformDeviate()
            gal = cosmos_cat.makeGalaxy(random.randint(0,cosmos_cat.nobjects-1), gal_type='parametric', chromatic=True, noise_pad_size = 0)

            gal = gal.rotate(ud() * 360. * galsim.degrees)
            redshift = gal.SED.redshift

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

            
            
            ############ LUMINOSITY ############# 
            # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
            bdgal_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures
            bdgal_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures
            bdgal_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures         
            
            
            
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
            galaxy_nir_noiseless = np.zeros((3,nir_stamp_size,nir_stamp_size))
            galaxy_vis_noiseless = np.zeros((1,vis_stamp_size,vis_stamp_size))
            galaxy_lsst_noiseless = np.zeros((6,lsst_stamp_size,lsst_stamp_size))
            galaxy_nir_noisy = np.zeros((3,nir_stamp_size,nir_stamp_size))
            galaxy_vis_noisy = np.zeros((1,vis_stamp_size,vis_stamp_size))
            galaxy_lsst_noisy = np.zeros((6,lsst_stamp_size,lsst_stamp_size))
            galaxy_noiseless = np.zeros((10,max_stamp_size,max_stamp_size))
            galaxy_noisy = np.zeros((10,max_stamp_size,max_stamp_size))

            i = 0
            for filter_name, filter_ in filters.items(): 
                if (i < 3):
                    poissonian_noise_nir = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_nir[i])
                    img = galsim.ImageF(nir_stamp_size, nir_stamp_size, scale=pixel_scale_euclid_nir)  
                    bdfinal = galsim.Convolve([bdgal_euclid_nir, PSF_euclid_nir])
                    bdfinal.drawImage(filter_, image=img)
                    # Noiseless galaxy
                    galaxy_nir_noiseless[i]= img.array.data
                    galaxy_noiseless[i,int((max_stamp_size/2)-(nir_stamp_size/2)):int((max_stamp_size/2)+(nir_stamp_size/2)),
                           int((max_stamp_size/2)-(nir_stamp_size/2)):int((max_stamp_size/2)+(nir_stamp_size/2))] = galaxy_nir_noiseless[i]
                    # Noisy galaxy
                    img.addNoise(poissonian_noise_nir)
                    galaxy_nir_noisy[i]= img.array.data
                    galaxy_noisy[i,int((max_stamp_size/2)-(nir_stamp_size/2)):int((max_stamp_size/2)+(nir_stamp_size/2)),
                           int((max_stamp_size/2)-(nir_stamp_size/2)):int((max_stamp_size/2)+(nir_stamp_size/2))] = galaxy_nir_noisy[i]
                else:
                    if (i==3):
                        poissonian_noise_vis = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_vis)
                        img = galsim.ImageF(vis_stamp_size, vis_stamp_size, scale=pixel_scale_euclid_vis)  
                        bdfinal = galsim.Convolve([bdgal_euclid_vis, PSF_euclid_vis])
                        bdfinal.drawImage(filter_, image=img)
                        # Noiseless galaxy
                        galaxy_vis_noiseless[3-i]= img.array.data
                        galaxy_noiseless[i,int((max_stamp_size/2)-(vis_stamp_size/2)):int((max_stamp_size/2)+(vis_stamp_size/2)),
                               int((max_stamp_size/2)-(vis_stamp_size/2)):int((max_stamp_size/2)+(vis_stamp_size/2))] = galaxy_vis_noiseless[3-i]
                        # Noisy galaxy
                        img.addNoise(poissonian_noise_vis)
                        galaxy_vis_noisy[3-i]= img.array.data
                        galaxy_noisy[i,int((max_stamp_size/2)-(vis_stamp_size/2)):int((max_stamp_size/2)+(vis_stamp_size/2)),
                               int((max_stamp_size/2)-(vis_stamp_size/2)):int((max_stamp_size/2)+(vis_stamp_size/2))] = galaxy_vis_noisy[3-i]
                    else:
        #                print('passage a LSST')
                        poissonian_noise_lsst = galsim.PoissonNoise(rng, sky_level_pixel_lsst[i-4])
                        img = galsim.ImageF(lsst_stamp_size, lsst_stamp_size, scale=pixel_scale_lsst)  
                        bdfinal = galsim.Convolve([bdgal_lsst, PSF_lsst])
                        bdfinal.drawImage(filter_, image=img)
                        # Noiseless galaxy
                        galaxy_lsst_noiseless[i-4]= img.array.data
                        galaxy_noiseless[i,int((max_stamp_size/2)-(lsst_stamp_size/2)):int((max_stamp_size/2)+(lsst_stamp_size/2)),
                               int((max_stamp_size/2)-(lsst_stamp_size/2)):int((max_stamp_size/2)+(lsst_stamp_size/2))] = galaxy_lsst_noiseless[i-4]
                        # Noisy galaxy
                        img.addNoise(poissonian_noise_lsst)
                        galaxy_lsst_noisy[i-4]= img.array.data
                        galaxy_noisy[i,int((max_stamp_size/2)-(lsst_stamp_size/2)):int((max_stamp_size/2)+(lsst_stamp_size/2)),
                               int((max_stamp_size/2)-(lsst_stamp_size/2)):int((max_stamp_size/2)+(lsst_stamp_size/2))] = galaxy_lsst_noisy[i-4]
                i+=1
            return galaxy_noiseless, galaxy_noisy, redshift
        except RuntimeError: 
            count +=1
    print("nb of error : "+(count))
    

    
    
    
    
    
    
# Generation function
def Gal_generator_noisy_pix_same(cosmos_cat):
    cosmos_cat = cosmos_cat
    count = 0
    galaxy = np.zeros((10))
    while (galaxy.all() == 0):
        try:
            path, filename = os.path.split('__file__')    
            datapath = galsim.meta_data.share_dir
            datapath2 = os.path.abspath(os.path.join(path,'/sps/lsst/users/barcelin/EUCLID_Filters/'))
            #print('passed here')
            # initialize (pseudo-)random number generator
            random_seed = 1234567
            rng = galsim.BaseDeviate(random_seed+1)

                    # read in the Euclid NIR filters
            filter_names_euclid_nir = 'HJY'
            filter_names_euclid_vis = 'V'

            # read in the LSST filters
            filter_names_lsst = 'ugrizy'
            filters = {}
            #print('passed here 2')

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

            # Chromatic bulge+disk galaxy
            # On prend la distribution de galaxies en fonction du redshift du Science Book de LSST
            i = 25.
            z0 = (0.0417*i)-0.744
            zl=np.arange(0.001,5,0.01)

            p_unnormed = lambda z : ((1/(2*z0))*((z/z0)**2)*np.exp(-z/z0))
            p_normalization = scipy.integrate.quad(p_unnormed, 0., np.inf)[0]
            p = lambda z : p_unnormed(z) / p_normalization

            from scipy import stats
            class LSST_redshift_distribution(stats.rv_continuous):
                def __init__(self):
                    super(LSST_redshift_distribution, self).__init__()
                    self.a = 0.
                    self.b = 10.
                def _pdf(self, x):
                    return p(x)

            nz = LSST_redshift_distribution()

            redshift = nz.rvs()   

            ############## SHAPE OF THE GALAXY ##################
            ud = galsim.UniformDeviate()
            gal = cosmos_cat.makeGalaxy(random.randint(0,cosmos_cat.nobjects-1), gal_type='parametric', chromatic=True, noise_pad_size = 0)

            gal = gal.rotate(ud() * 360. * galsim.degrees)
            redshift = gal.SED.redshift

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

            
            
            ############ LUMINOSITY ############# 
            # The luminosity is multiplied by the ratio of the noise in the LSST R band and the assumed cosmos noise             
            bdgal_lsst =  (15. * (6.68**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures
            bdgal_euclid_nir =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures
            bdgal_euclid_vis =  (1800. * ((1.25)**2 - (0.37)**2)/((2.4**2)*(1.-0.33**2))) * gal * N_exposures         
            
            
            
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

            i = 0
            for filter_name, filter_ in filters.items(): 
                if (i < 3):
                    poissonian_noise_nir = galsim.PoissonNoise(rng, sky_level=sky_level_pixel_nir[i])
                    img = galsim.ImageF(max_stamp_size,max_stamp_size, scale=pixel_scale_euclid_nir)  
                    bdfinal = galsim.Convolve([bdgal_euclid_nir, PSF_euclid_nir])
                    bdfinal.drawImage(filter_, image=img)
                    # Noiseless galaxy
                    galaxy_nir_noiseless[i]= img.array.data
                    galaxy_noiseless[i] = galaxy_nir_noiseless[i]
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
                        # Noiseless galaxy
                        galaxy_lsst_noiseless[i-4]= img.array.data
                        galaxy_noiseless[i] = galaxy_lsst_noiseless[i-4]
                        # Noisy galaxy
                        img.addNoise(poissonian_noise_lsst)
                        galaxy_lsst_noisy[i-4]= img.array.data
                        galaxy_noisy[i]= galaxy_lsst_noisy[i-4]
                i+=1
            return galaxy_noiseless, galaxy_noisy, redshift
        except RuntimeError: 
            count +=1
        print("nb of error : "+str(count))
