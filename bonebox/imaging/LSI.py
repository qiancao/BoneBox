# -*- coding: utf-8 -*-

"""

imaging.LSI

Linear Shift-invariant Imaging Models of Blur and Noise

A Fourier-domain implementation

Author:  Qian Cao

"""

import numpy as np

def getFrequencyAxis(image, spacing):
    # spacing: voxel or pixel size (mm)
    # image: only used to retrieve shape
    # returns list of frequencies
    
    freq = []
    
    N = image.shape
    if len(N) != len(spacing):
        raise(ValueError, "Dimension of image and spacing must match")
    
    for dim in range(N):
        freq.append(np.fft.fftfreq(N[dim], spacing[dim]))
        
    return freq

def imageFFT2D(image, spacing):
    # apply fft on projection images
    # spacing : pixel extent
    
    freq = getFrequencyAxis(image, spacing)
    imageFFT = np.fft.fft2(image)
    
    return freq, imageFFT

def imageFFT3D(image, spacing):
    # apply fft on reconstructed images
    # spacing : voxel or extent
    
    freq = getFrequencyAxis(image, spacing)
    imageFFT = np.fft.fftn(image)
    
    return freq, imageFFT

def rebin2D(image, pixelSize, pixelSizeTarget, origin=(0,0)):
    pass

def rebin3D(image, voxelSize, voxelSizeTarget, origin=(0,0,0)):
    pass

def makeGaussMTF2D(F):
    pass

def makeGaussNPS2D(F):
    pass

def makeGaussMTF3D():
    pass

def makeApodizedGaussNPS3D():
    pass

def applyMTF2D(image, F, MTF):
    pass

def applyNPS2D(image, F, MTF, randState):
    pass

def applyMTF3D(image, F, MTF):
    pass

def applyNPS3D(image, F, MTF, randState):
    pass


if __name__ == "__main__":
    pass
