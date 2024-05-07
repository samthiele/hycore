from gfit import mgauss
import numpy as np
import hylite
import skimage.data as skdata
from skimage.transform import rotate

from hylite.reference import genImage, randomSpectra

def genTray(ncores=3, flip=False, masked=False):
    """
    :param ncores: Number of core sections to include in this tray.
    :param flip: True if dominant vs accessory class should be flipped. Can be used to define different lithologies.
    :param mask: True if tray pixels should be replaced with nan. Default is False.
    :return: hsi (hylite.HyImage) spectra, A (hylite.HyImage) abundances and a core mask (np.ndarray).
    """

    # get a the random image
    im, A = genImage(flip=flip)
    im.data = im.data[:, int(im.ydim() * 0.3):int(im.ydim() * 0.7), : ] # crop a bit so that box is not square!

    # generate a box mask
    boxmask = np.full( (im.xdim(), im.ydim()), False)
    boxmask[10:-10, :] = True
    for s in np.linspace(0, im.ydim(), ncores + 1):
        boxmask[:, max(int(s) - 5, 0):min(int(s + 5), im.ydim())] = False

    # add asymmetric bit to mask (useful for checking)
    boxmask[0:100,50:100] = False
    
    # apply box mask
    eK = randomSpectra(im.get_wavelengths(), f=[2100., 2245.], d=[0.1, 0.05], a=0.1)
    im.data[ ~boxmask, : ] = eK[None,:]

    if masked:
        A.data[~boxmask] = np.nan
        im.data[~boxmask] = np.nan
    return im, A, boxmask
