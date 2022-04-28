# -*- coding: utf-8 -*-

"""

imaging.LSI

Linear Shift-invariant Imaging Models of Blur and Noise

A Fourier-domain implementation

See test_20220419_bin for draft implementation.

Author:  Qian Cao

"""

import nrrd
import numpy as np
import os, sys
from scipy import ndimage
import matplotlib.pyplot as plt
import os

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
    
    # TODO: Validate this for ROIs
    
    ndim = rois[0].ndim
    shape = rois[0].shape
    
    freqs = []
    for ind, Nvoxels in enumerate(shape):
        freqs.append(np.fft.fftfreq(Nvoxels,voxSize[ind]))
    
    N = len(rois)
    # A = np.prod(voxSize)
    
    Froi = np.zeros(shape,dtype=np.float64)
    for n, roi in enumerate(rois):
         Froi = Froi + np.abs(np.fft.fftn(roi, axes=tuple(range(ndim))))
    
    # NPS = Froi / N / A
    NPS = np.prod(voxSize) / np.prod(shape) * (Froi / N)
    
    return NPS, freqs

def applyMTF(image, MTF):
    # Apply blur to image
    
    # assumes MTF axes is consistent with image
    assert image.shape == MTF.shape
    
    imageFFT = np.fft.fftn(image,axes=tuple(range(image.ndim)))
    image_filtered = abs(np.fft.ifftn(imageFFT * MTF, axes=tuple(range(image.ndim))))
    
    return image_filtered

def applySampling(image, voxSize, voxSizeNew):
    # TODO: check on use of histogramdd
    
    dimsNew = (image.shape * np.array(voxSize) // np.array(voxSizeNew)).astype(int)
    
    shape = image.shape
    # xxx,yyy,zzz = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]),np.arange(shape[2]))
    # TODO: think about how to do this properly
    x,y,z = np.meshgrid(np.linspace(0,shape[0],dimsNew[0]).astype(int),
                                 np.linspace(0,shape[1],dimsNew[1]).astype(int),
                                 np.linspace(0,shape[2],dimsNew[2]).astype(int))

    imageNew = ndimage.map_coordinates(image, [x.flatten(), y.flatten(), z.flatten()])
    
    imageNew = np.reshape(imageNew,dimsNew)
    
    # return ndimage.zoom(image, zoom, prefilter=False)
    return imageNew

def noisePoisson(size,plambda=1000,seed=None):
    rng = np.random.default_rng(seed)
    return rng.poisson(plambda,size) - plambda

def noiseNormal(size,mean=0,std=1,seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(mean,std,size)

def getFreqs(shape,voxSize):
    freqs = []
    for ind, Nvoxels in enumerate(shape):
        freqs.append(np.fft.fftfreq(Nvoxels,voxSize[ind]))
    return freqs

def Gaussian3DUncorrelated(xxx,yyy,zzz,std):
    return np.exp((-xxx**2/std[0]**2-yyy**2/std[1]**2-zzz**2/std[2]**2)/2)

def make3DMTFGaussian(freqs, std):
    xxx,yyy,zzz = np.meshgrid(*freqs)
    return Gaussian3DUncorrelated(xxx,yyy,zzz,std)

def shapeNPS(noise, NPS):
    # deprecated
    
    # assumes NPS axes is consistent with image
    assert noise.shape == NPS.shape
    
    noiseFFT = np.fft.fftn(noise**2,axes=tuple(range(noise.ndim)))
    noise_filtered = np.abs(np.fft.ifftn(noiseFFT * np.sqrt(NPS), axes=tuple(range(noise.ndim))))
    
    return noise_filtered

def NPS2noise(NPS,seed=None):
    # generates noise from a white noise and power spectra
    # note: frequency axes must match image
    # see https://www.mathworks.com/matlabcentral/fileexchange/36462-noise-power-spectrum
    
    rng = np.random.default_rng(seed)
    v = rng.random(NPS.shape,dtype=np.float64)
    F = NPS * (np.cos(2*np.pi*v) + 1j*np.sin(2*np.pi*v))
    f = np.fft.ifftn(F, axes=tuple(range(NPS.ndim)))
    noise = np.real(f) + np.imag(f)
    
    return noise

def makeNullCone(freqs,alpha=1,beta=0.5):
    xxx,yyy,zzz = np.meshgrid(*freqs)
    cone = alpha * np.sqrt(xxx**2+yyy**2)
    cone = np.maximum(cone - beta * abs(zzz),0)
    return cone

def makeNPSRamp(freqs,alpha=1):
    xxx,yyy,zzz = np.meshgrid(*freqs)
    cone = alpha * np.sqrt(xxx**2+yyy**2)
    return cone

def integrateNPS(NPS,freqs):
    df = 1
    for ind, f in enumerate(freqs):
        df = df * np.abs(f[1]-f[0]) # integrate noise power
    return np.sum(NPS) * df

if __name__ == "__main__":
    # TODO: Make example script
    pass