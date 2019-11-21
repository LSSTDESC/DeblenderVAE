import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

def plot_rgb(gal, bands=[5,6,7], ax=None, band_first=True, zoom=1.5, shifts=None):
    if ax is None:
        ax = plt.subplot()
    if band_first:
        tr = [1,2,0]
    else:
        tr = [0,1,2]
    
    imsize = float(gal.shape[1]) / 2.
    ax.imshow(np.clip(gal[:,:,:].transpose(tr)[:,:,bands], a_min=0.0, a_max=1.), extent=(-imsize,imsize,-imsize,imsize), origin='lower left')#0., 1. 
    if shifts is not None:
        for (x,y) in shifts:
            ax.scatter(x, y,  marker='+', c='r')
    ax.scatter(0., 0., marker='+', c='b')
    ax.set_xlim(-imsize/zoom,imsize/zoom)
    ax.set_ylim(-imsize/zoom,imsize/zoom)
    ax.axis('off')

def plot_all_bands(gal, band_first=True, cmap=mpl.cm.gray, zoom=1.5):
    filters = 'HJYVugrizy'
    bax = 0 if band_first else -1
    n = gal.shape[bax]
    fig, axes = plt.subplots(1, n, figsize=(4*n,4))
    imsize = float(gal.shape[1]) / 2.
    for i in range(n):
        ax = axes[i]
        if band_first:
            ax.imshow(np.clip(gal[i], 0., 1.), cmap=cmap, extent=(-imsize,imsize,-imsize,imsize))
        else:
            ax.imshow(np.clip(gal[:,:,i], 0., 1.), cmap=cmap, extent=(-imsize,imsize,-imsize,imsize))
        ax.axis('off')
        ax.set_xlim(-imsize/zoom,imsize/zoom)
        ax.set_ylim(-imsize/zoom,imsize/zoom)
        ax.set_title(filters[i])

# plot function for RGB image with the 6 LSST bandpass filters
def plot_rgb_lsst(ugrizy_img, stamp_size, ax=None):
    RGB_img = np.zeros((stamp_size,stamp_size,3))
    if ax is None:
        _, ax = plt.subplots(1,1)
    max_img = np.max(ugrizy_img)

    coeff = int(len(ugrizy_img[0])/stamp_size) - 1

    ugrizy_img = ugrizy_img[:,:,:]
    RGB_img[:,:,0] = ugrizy_img[0+16*coeff:64-16*coeff,0+16*coeff:64-16*coeff,1]
    RGB_img[:,:,1] = ugrizy_img[0+16*coeff:64-16*coeff,0+16*coeff:64-16*coeff,2]
    RGB_img[:,:,2] = ugrizy_img[0+16*coeff:64-16*coeff,0+16*coeff:64-16*coeff,3]
    ax.imshow(np.clip(RGB_img[:,:,[0,1,2]], a_min=0.0, a_max=None) / max_img, origin='lower left')

# plot function for RGB image with the 10 LSST+Euclid bandpass filters
def plot_rgb_lsst_euclid(ugrizy_img, stamp_size, ax=None):
    RGB_img = np.zeros((stamp_size,stamp_size,3))
    if ax is None:
        _, ax = plt.subplots(1,1)
    max_img = np.max(ugrizy_img[:,:,4:])

    coeff = int(len(ugrizy_img[0])/stamp_size) - 1

    ugrizy_img = ugrizy_img[:,:,:]
    RGB_img[:,:,0] = ugrizy_img[0+16*coeff:64-16*coeff,0+16*coeff:64-16*coeff,5]
    RGB_img[:,:,1] = ugrizy_img[0+16*coeff:64-16*coeff,0+16*coeff:64-16*coeff,6]
    RGB_img[:,:,2] = ugrizy_img[0+16*coeff:64-16*coeff,0+16*coeff:64-16*coeff,8]
    ax.imshow(np.clip(RGB_img[:,:,[2,1,0]], a_min=0.0, a_max=None) / max_img, origin='lower left')


# Plot galaxies on single band and scatter number on each galaxies
def scatter_galaxies(image, shift, pixel_scale, stamp_size, scatter = 'numbers', blendedness = None, ax=None):
    """
    Parameters:
    ----------
    image: single band image
    shift: list of shift (output from blended images generation function)
    pixel_scale: pixel scale of the bandpass filter used to plot the image
    stamp_size: size of the stamp
    """
    ax.imshow(image)

    if scatter == 'numbers':
        for k in range (len(shift)):
            ax.scatter((stamp_size) + shift[k][0]/pixel_scale, (stamp_size) + shift[k][1]/pixel_scale, s = 50 ,c='red', marker="${}$".format(k))
    elif scatter == 'blendedness':
        for k in range (len(blendedness)):
            ax.scatter((stamp_size) + shift[k][0]/pixel_scale, (stamp_size) + shift[k][1]/pixel_scale, s = 500 ,c='red', marker="${0:.2f}$".format(blendedness[k]))
    #elif scatter == 'None':
            #ax.scatter((stamp_size) + shift[k][0]/pixel_scale, (stamp_size) + shift[k][1]/pixel_scale, s = 500 ,c='red', '.')


# Function to compute mean and variance in each bins of histograms
def mean_var(x,y,bins):
    """
    Return mean and variance in each bins of the histogram
    """
    n,_ = np.histogram(x,bins=bins, weights=None)
    ny,_ = np.histogram(x,bins=bins, weights=y)
    mean_y = ny/n
    ny2,_ = np.histogram(x,bins=bins, weights=y**2)
    var_y = (ny2/n - mean_y**2)/n
    
    return (mean_y, var_y)

# Function to create a circular mask around the center of the galaxy
def createCircularMask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask




# Plot error on variable as function of a specific parameter
# def add_images_to_plot(p, df, axes, images, y_height, nb_of_images = 3):
#     """
#     Permit to add blended images on top of the plot wanted
#     """
#     idx = []
#     idx.append(np.random.choice(np.where( (df[p]>0.45*np.max(df[p])) & (df[p]<0.55*np.max(df[p])) )[0], size = 1, replace = False)[0])
#     idx.append(np.random.choice(np.where(df[p]<0.25*np.max(df[p]))[0], size = 1, replace = False)[0])
#     idx.append(np.random.choice(np.where( (df[p]>0.8*np.max(df[p])) & (df[p]<0.85*np.max(df[p])) )[0], size = 1, replace = False)[0])
    
#     y_lim = y_height
#     for i in range (nb_of_images):
#         image = images[idx[i]]
#         imagebox = OffsetImage(image[1][6], zoom=2)
#         ab = AnnotationBbox(imagebox, (df[p][idx[i]], y_lim))
#         axes.add_artist(ab)
#         axes.text(df[p][idx[i]]-0.07, y_lim - np.abs(y_height/1.5), 'blend rate = '+str(np.round(df[p][idx[i]], 2)), fontsize =14, color='r')

# def v_as_function_of_p(v, p, v_labels, x_label, y_label, bins = 50, x_log_scale = False, y_log_scale = False, variance = False, xlim= None, 
#                        ylim = None, add_images = False, y_height = None, parameter = None, images= None, df = None):
#     """
#     Plot the variable(s) v as function of the parameter p.

#     Parameters:
#     ----------
#     v: variable(s) to plot. Insert it as a list of numpy array.
#     p: parameter to use. Insert it as a list of numpy array.
#     v_labels: label for each variable
#     x_label: label for the parameter used
#     y_label: label for the variable used
#     """
    
#     font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 22}
#     matplotlib.rc('font', **font)
#     fig, axes = plt.subplots(2, figsize=(16,14), sharex = True)
#     fig.subplots_adjust(right=1, left=0,hspace=0,wspace=0.1)
    
#     x = np.linspace(np.min(p[0]), np.max(p[0]), len(bins)-1)
#     mid = (bins[1:]+bins[:-1])*0.5
       
#     axes[0].hist(p[0], bins = bins, alpha = 0.4)
#     axes[0].set_title('Distribution of '+x_label)
#     axes[0].set_yticklabels([])
#     axes[0].set_xticklabels([])
    
#     if add_images:
#         add_images_to_plot(parameter, df, axes, images, y_height)

#     mean = []
#     var = []
#     for i in range (len(p)):
#         p_is_nan = np.where(np.isnan(p[i]))[0]
#         v[i] = np.delete(v[i], p_is_nan)
#         p[i] = np.delete(np.array(p[i]), p_is_nan)

#         mean.append(mean_var(p[i],v[i], bins = np.log10(bins))[0])
#         var.append(mean_var(p[i],v[i], bins = np.log10(bins))[1])

#         axes[1].plot(mid,mean[i], label = v_labels[i])
#         if variance == True:
#             axes[1].fill_between(mid, mean[i] - 1*var[i]**0.5, mean[i] + 1*var[i]**0.5, alpha=0.5)
#             #print('Warning: variance is augmented 10 times')
        
#     axes[1].plot(np.arange(0,1000), np.zeros(1000))
#     if xlim != None:
#         axes[1].set_xlim(xlim)
#     else: 
#         axes[1].set_xlim(np.min(p[0]), np.max(p[0]))
#     if ylim != None:
#         axes[1].set_ylim(ylim)
#     else:
#         axes[1].set_ylim(np.min(v[0]), np.max(v[0]))
#     if x_log_scale:
#         axes[0].set_xscale('log')
#         axes[1].set_xscale('log')
#     if y_log_scale:
#         axes[1].set_yscale('log')
#     axes[1].set_xlabel(x_label)
#     axes[1].set_ylabel(y_label)
#     axes[1].legend(loc = "upper right")

#     return axes


# def v_as_function_of_p_2(v, p, v_labels, x_label, y_label, bins = 50, x_log_scale = False, y_log_scale = False, variance = False, xlim= None, 
#                        ylim = None, add_images = False, y_height = None, parameter = None, images= None, df = None):
#     """
#     Plot the variable(s) v as function of the parameter p.

#     Parameters:
#     ----------
#     v: variable(s) to plot. Insert it as a list of numpy array.
#     p: parameter to use. Insert it as a list of numpy array.
#     v_labels: label for each variable
#     x_label: label for the parameter used
#     y_label: label for the variable used
#     """
    
#     font = {'family' : 'normal',
#         'weight' : 'normal',
#         'size'   : 22}
#     matplotlib.rc('font', **font)
#     fig, axes = plt.subplots(1, figsize=(16,8), sharex = True)
#     fig.subplots_adjust(right=1, left=0,hspace=0,wspace=0.1)
    
#     x = np.linspace(np.min(p[0]), np.max(p[0]), bins)
#     mid = (x[0:]+x[:])*0.5
           
#     if add_images:
#         add_images_to_plot(parameter, df, axes, images, y_height)

#     mean = []
#     var = []
#     for i in range (len(p)):
#         p_is_nan = np.where(np.isnan(p[i]))[0]
#         v[i] = np.delete(v[i], p_is_nan)
#         p[i] = np.delete(np.array(p[i]), p_is_nan)

#         mean.append(mean_var(p[i],v[i], bins = bins)[0])
#         var.append(mean_var(p[i],v[i], bins = bins)[1])

#         axes.plot(mid,mean[i], label = v_labels[i])
#         if variance == True:
#             axes.fill_between(mid, mean[i] - 1*var[i]**0.5, mean[i] + 1*var[i]**0.5, alpha=0.5)
#             #print('Warning: variance is augmented 10 times')
        
#     axes.plot(np.arange(0,1000), np.zeros(1000))
#     if xlim != None:
#         axes.set_xlim(xlim)
#     else: 
#         axes.set_xlim(np.min(p[0]), np.max(p[0]))
#     if ylim != None:
#         axes.set_ylim(ylim)
#     else:
#         axes.set_ylim(np.min(v[0]), np.max(v[0]))
#     if x_log_scale:
#         axes.set_xscale('log')
#     if y_log_scale:
#         axes.set_yscale('log')
#     axes.set_xlabel(x_label)
#     axes.set_ylabel(y_label)
#     axes.legend(loc = "upper right")

#     return axes

# To plot corner plot of latente space
def plot_corner_latent(z, lim=3, nbins=25, show_title=True):
    """
    Make a corner plot of standard gaussian distributed latent variables.
    Parameters
    ----------
    z : latent variables, array of size (n_samples, latent_dim)
    lim : int, optional
        [description], by default 3
    nbins : int, optional
        [description], by default 25
    show_title : bool, optional
        [description], by default True
    Example
    -------
    z = np.random.normal(size=(1000,8))
    plot_corner_latent(z)
    """

    import matplotlib as mpl
    from mpl_toolkits.axes_grid1 import Grid
    
    latent_dim = z.shape[1]
    
    fig = plt.figure(figsize=(latent_dim*2,latent_dim*2))
    grid = Grid(fig, rect=111, nrows_ncols=(latent_dim,latent_dim), axes_pad=0.25, label_mode='L', share_y=False)
    
    colors = mpl.cm.jet(np.linspace(0,1,latent_dim))
    bins = nbins#np.linspace(-lim,+lim,nbins)
    
    for i in range(latent_dim):
        for j in range(latent_dim):
            ax = grid[i*latent_dim+j]
            if i == j :
                n,_,_ = ax.hist(z[:,i], bins=bins, normed=True, color=colors[i])
                ax.set_yticks([])
                if show_title:
                    ax.set_title('$z_{}$'.format(i))
            if i > j :
                ax.hist2d(z[:,j], z[:,i], bins=bins, cmap=mpl.cm.gray)
            if i < j :
                ax.axis('off')
    
    plt.tight_layout()
    
    #return fig