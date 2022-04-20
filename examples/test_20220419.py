# -*- coding: utf-8 -*-
"""
@author: Qian.Cao
"""

import nrrd
import numpy as np
import os, sys
from scipy import ndimage
import matplotlib.pyplot as plt

def computeNPS(rois,voxSize):
    """
    Compute noise power spectra from ROIs

    Parameters
    ----------
    rois : list of numpy arrays (must be the same size)
        list of noise ROIs.
    voxSize : tuple
        voxel dimension in the corresponding directions.

    Returns
    -------
    NPS: noise power spectra
    freqs: frequency vector for each axis

    """
    
    ndim = rois[0].ndim
    shape = rois[0].shape
    
    freqs = []
    for ind, Nvoxels in enumerate(shape):
        freqs.append(np.fft.fftfreq(Nvoxels,voxSize[ind]))
    
    N = len(rois)
    A = np.prod(voxSize)
    
    Froi = np.zeros(shape,dtype=np.complex128)
    for n, roi in enumerate(rois):
         Froi += np.fft.fftn(roi, axes=tuple(range(ndim)))
    
    NPS = np.abs(Froi) / N / A
    
    return NPS, freqs

def applyMTF(image, voxSize, MTF, freqs):
    
    return image

def applyBinning(image, voxSize, binSize):
    
    return image

def noisePoisson(size,plambda=1000,seed=None):
    rng = np.random.default_rng(seed)
    return rng.poisson(plambda,size) - plambda

def noiseNormal(size,mean=0,std=1,seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(mean,std,size)

if __name__ == "__main__":
    
    voxelSize = (0.05, 0.05, 0.05) # mm
    
    filenameNRRD = "../data/rois/isodata_04216_roi_4.nrrd"
    roi, header = nrrd.read(filenameNRRD)
    
    noise_rois = []
    for ind in range(200):
        noise = noiseNormal(roi.shape)
        noise_filt = ndimage.gaussian_filter(noise,1)
        noise_rois.append(noise_filt)
    
    NPS, freqs = computeNPS(noise_rois,voxelSize)
    NPS0, freqs = computeNPS([noise_rois[0]],voxelSize)
    
    plt.figure()    
    plt.close("all")
    plt.plot(freqs[0],NPS[:,0,0])
    plt.plot(freqs[0],NPS0[:,0,0])
    
    
    