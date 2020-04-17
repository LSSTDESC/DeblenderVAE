from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

from . import layers

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy
from IPython.display import clear_output, set_matplotlib_formats
from mpl_toolkits.axes_grid1 import make_axes_locatable

def build_vanilla_vae(encoder, decoder, coeff_KL,full_cov=False):
    """
    Returns the model to train and parameters to plot relevant information during training using the VAEHistory callback
    """
    input_vae = Input(shape=encoder.input.shape[1:])
    output_encoder = encoder(input_vae)

    z, Dkl = layers.SampleMultivariateGaussian(full_cov=full_cov, add_KL=False, return_KL=True, coeff_KL=coeff_KL)(output_encoder)
    
    vae = Model(input_vae, decoder(z))
    vae_utils = Model(input_vae, [*encoder(input_vae), z, Dkl, decoder(z)])

    return vae, vae_utils, Dkl



class VAEHistory(Callback):
    def __init__(self, xval_sub, vae_utils, latent_dim, alpha, plot_bands=0, figroot=None, period=1):
        self.xval_sub = xval_sub
        self.latent_dim = latent_dim
        self.plot_bands = plot_bands
        self.vae_utils = vae_utils
        self.figroot = figroot
        
        self.counter = 0
        self.period = period
        
        self.epoch = 0
        self.loss = []
        self.val_loss = []
        self.D_KL = []
        
        self.zz = np.linspace(-5,5,1000)
        self.gauss = scipy.stats.norm.pdf(self.zz)
                
        self.colors = mpl.cm.jet(np.linspace(0,1,latent_dim))
        self.alpha = alpha

        self.counter = 0
        
    def mask_outliers(points_plop, thresh=3.5):
        """
        Returns a boolean array with True if points are outliers and False 
        otherwise.

        Parameters:
        -----------
            points : An numobservations by numdimensions array of observations
            thresh : The modified z-score to use as a threshold. Observations with
                a modified z-score (based on the median absolute deviation) greater
                than this value will be classified as outliers.

        Returns:
        --------
            mask : A numobservations-length boolean array.

        References:
        ----------
            Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
            Handle Outliers", The ASQC Basic References in Quality Control:
            Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
        """
        points = np.array(points_plop)

        if len(points.shape) == 1:
            points = points[:,None]
        median = np.median(points, axis=0)
        diff = np.sum((points - median)**2, axis=-1)
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)

        modified_z_score = 0.6745 * diff / med_abs_deviation

        return np.ma.masked_array(data=points, mask=modified_z_score > thresh)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1
        self.counter +=1

        if self.counter % self.period == 0:
            self.loss.append(logs.get('loss'))
            self.val_loss.append(logs.get('val_loss'))
                    
            mu, sigma, z, dkl, out = self.vae_utils.predict(self.xval_sub[:,1])

            self.D_KL.append(np.mean(dkl))

            clear_output(wait=True)
            
            fig, axes = plt.subplots(2, 4, figsize=(4*4,2*4))
            axes = axes.ravel()
            
            # Plot loss
            ax = axes[0]
            ax.plot(self.loss, c='b', label='train')
            ax.plot(self.val_loss, c='r', label='val')
            ax.legend()
            ax.set_xlabel('epoch')
            ax.set_title('Training/validation losses')
            ax.set_yscale('log')
            
            # Plot DKL
            ax = axes[1]
            ax.plot(self.D_KL)
            ax.set_xlabel('epoch')
            ax.set_title('D_KL')
            ax.set_yscale('log')

            # Plot mu_z, sigma_z and z
            for dim in range(self.latent_dim):
                c = self.colors[dim]
                _ = axes[2].hist(mu[:,dim], bins=50, histtype='step', color=c, label=str(dim), density=True)
                _ = axes[3].hist(sigma[:,dim], bins=50, log=False, histtype='step', color=c, label=str(dim), density=True)
                _ = axes[4].hist(z[:,dim], bins=50, histtype='step', color=c, label=str(dim), density=True)
            
            axes[4].plot(self.zz, self.gauss, c='k')
            axes[4].set_xlim(-5,5)
            
            axes[2].set_title('mu_y(X)')
            axes[3].set_title('sigma_y(X)')
            axes[4].set_title('z~N(0,1)')
            
            idx = np.random.randint(len(self.xval_sub))
            # weird thing when indexing numpy arrays with list...
            if isinstance(self.plot_bands, int):
                img_in = self.xval_sub[idx,1,:,:,self.plot_bands]
                print('first'+str(img_in.shape))
                img_target = self.xval_sub[idx,0,:,:,self.plot_bands]
                img_out = out[idx,:,:,self.plot_bands]
            else:
                img_in = self.xval_sub[idx,1,:,:,self.plot_bands].transpose([1,2,0])
                print('second'+str(img_in.shape))
                img_target = self.xval_sub[idx,0,:,:,self.plot_bands].transpose([1,2,0])
                img_out = out[idx,:,:,self.plot_bands].transpose([1,2,0])
            
            # Plot noisy input
            ax = axes[5]
            im = ax.imshow(np.clip(img_in[:,:,0], 0., 1.))
            if isinstance(self.plot_bands, int):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            
            ax.set_title('input')
            ax.axis('off')

            # Plot noiseless target
            ax = axes[6]
            im = ax.imshow(np.clip(img_target[:,:,0], 0., 1.))
            if isinstance(self.plot_bands, int):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            
            ax.set_title('target')
            ax.axis('off')
            
            # Plot networks output
            ax = axes[7]
            im = ax.imshow(np.clip(img_out[:,:,0], 0., 1.))
            if isinstance(self.plot_bands, int):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                plt.colorbar(im, cax=cax)
            ax.set_title('output')
            ax.axis('off')
            
            plt.tight_layout()
            
            if self.figroot is not None:
                fig.savefig(self.figroot+"_epoch"+str(self.epoch)+".png")

            self.counter = 0
