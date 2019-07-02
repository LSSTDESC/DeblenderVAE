import numpy as np
import matplotlib.pyplot as plt

# plot function for RGB image with the 6 LSST bandpass filters
def plot_rgb_lsst(ugrizy_img, stamp_size, ax=None):
    RGB_img = np.zeros((stamp_size,stamp_size,3))
    if ax is None:
        _, ax = plt.subplots(1,1)
    max_img = np.max(ugrizy_img)
    ugrizy_img = ugrizy_img[:,:,:].reshape((6,stamp_size,stamp_size))
    RGB_img[:,:,0] = ugrizy_img[1][:,:]
    RGB_img[:,:,1] = ugrizy_img[2][:,:]
    RGB_img[:,:,2] = ugrizy_img[4][:,:]
    ax.imshow(np.clip(RGB_img[:,:,[2,1,0]], a_min=0.0, a_max=None) / max_img)

# plot function for RGB image with the 10 LSST+Euclid bandpass filters
def plot_rgb_lsst_euclid(ugrizy_img, stamp_size, ax=None):
    RGB_img = np.zeros((stamp_size,stamp_size,3))
    if ax is None:
        _, ax = plt.subplots(1,1)
    max_img = np.max(ugrizy_img[4:])
    ugrizy_img = ugrizy_img[:,:,:].reshape(10,stamp_size,stamp_size)
    RGB_img[:,:,0] = ugrizy_img[5][:,:]
    RGB_img[:,:,1] = ugrizy_img[6][:,:]
    RGB_img[:,:,2] = ugrizy_img[8][:,:]
    ax.imshow(np.clip(RGB_img[:,:,[2,1,0]], a_min=0.0, a_max=None) / max_img)


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