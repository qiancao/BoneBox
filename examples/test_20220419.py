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

def applySampling(image, voxSize, zoom):
    # binSize
    
    # newDims = (image.shape * np.array(voxSize) // np.array(binSize)).astype(int)
    # zoom = np.array(binSize) / np.array(voxSize)
    # TODO: transition to binning using histogramdd

    return ndimage.zoom(image, zoom, order=3, prefilter=True)

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
        df = df * np.abs(f[1]-f[0])
    
    return np.sum(NPS) * df

if __name__ == "__main__":
    
    outDir = "/gpfs_projects/qian.cao/BoneBox-out/test_20220419/"
    
    plt.close("all")
    
    voxSize = (0.05, 0.05, 0.05) # mm
    boneHU = 400 # HU
    
    noise_std = 250
    
    stdMTF = [1,1,0.5]
    
    filenameNRRD = "../data/rois/isodata_04216_roi_4.nrrd"
    roi, header = nrrd.read(filenameNRRD)
    
    # normalize to 0 and 1, scale to bone HU
    roi = (roi - np.min(roi)) / (np.max(roi) - np.min(roi))
    roi = roi * boneHU
    
    freqs = getFreqs(roi.shape,voxSize)
    T = make3DMTFGaussian(freqs,stdMTF)
    roiT = applyMTF(roi,T)
    
    ramp = makeNPSRamp(freqs)
    # noise = noiseNormal(roi.shape,mean=0,std=stdNoise,seed=None)
    # S = (T**2) * cone
    # noiseS = applyMTF(noise, S)
    # noiseS = np.sqrt(noiseS / np.sum(noiseS) * noise_std**2)
    S = (T**2)*ramp
    S = S / integrateNPS(S,freqs) * noise_std # TODO: should be squqred
    
    noiseS = NPS2noise(S,seed=None) * (noise_std**2)
    
    # noiseS = ndimage.gaussian_filter(noise,5) * noise_std
    
    #%% Preview MTF and NPS
    
    # coneshift = np.fft.fftshift(cone)
    Tshift = np.fft.fftshift(T)
    Sshift = np.fft.fftshift(S)
    freqx = np.fft.fftshift(freqs[0])
    
    roi_final = roiT+noiseS
    
    axlims = [np.min(freqs[0]), np.max(freqs[0]), 
              np.min(freqs[1]), np.max(freqs[1]),
              np.min(freqs[2]), np.max(freqs[2])]
    
    fig, ax = plt.subplots(2,2,figsize=(12,10))
    img = ax[0,0].imshow(Tshift[:,:,101],cmap="viridis",extent=axlims[0:4])
    ax[0,0].set_xlabel("X (1/mm)")
    ax[0,0].set_ylabel("Y (1/mm)")
    cbar = fig.colorbar(img, ax=ax[0,0])
 
    img = ax[1,0].imshow(Tshift[:,101,:].T,cmap="viridis",extent=[np.min(freqs[0]), np.max(freqs[0]),np.min(freqs[2]), np.max(freqs[2])])
    ax[1,0].set_xlabel("X (1/mm)")
    ax[1,0].set_ylabel("Z (1/mm)")
    cbar = fig.colorbar(img, ax=ax[1,0])
    
    img = ax[0,1].imshow(Sshift[:,:,101],cmap="gnuplot2",extent=axlims[0:4])
    ax[0,1].set_xlabel("X (1/mm)")
    ax[0,1].set_ylabel("Y (1/mm)")
    cbar = fig.colorbar(img, ax=ax[0,1])
    cbar.set_label('HU^2 mm^3', rotation=90)
    
    img = ax[1,1].imshow(Sshift[:,101,:].T,cmap="gnuplot2",extent=[np.min(freqs[0]), np.max(freqs[0]),np.min(freqs[2]), np.max(freqs[2])])
    ax[1,1].set_xlabel("X (1/mm)")
    ax[1,1].set_ylabel("Z (1/mm)")
    cbar = fig.colorbar(img, ax=ax[1,1])
    cbar.set_label('HU^2 mm^3', rotation=90)
    
    plt.savefig(outDir+"3D Metrics.png")
    
    #%% 
    
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    
    ax[0].plot(freqx,Tshift[:,100,100],'k')
    ax[0].set_xlabel("Freqency (1/mm)")
    ax[0].set_xlim([0,np.max(freqx)])
    ax[0].set_title("In-plane MTF")
    
    ax[1].plot(freqx,Sshift[:,100,100],'k')
    ax[1].set_xlabel("Freqency (1/mm)")
    ax[1].set_ylabel("(HU^2 mm^3)")
    ax[1].set_xlim([0,np.max(freqx)])
    ax[1].set_title("In-plane NPS")
    
    plt.savefig(outDir+"3D Metrics in-plane.png")
    
    #%% Image Preview
    fig, ax = plt.subplots(2,2,figsize=(12,10))
    
    ax = ax.flatten()
    img = ax[0].imshow(roi[:,:,101],cmap="gray")
    ax[0].set_axis_off()
    cbar = fig.colorbar(img, ax=ax[0])
    cbar.set_label('HU', rotation=90)
    
    img = ax[1].imshow(roiT[:,:,101],cmap="gray")
    ax[1].set_axis_off()
    cbar = fig.colorbar(img, ax=ax[1])
    cbar.set_label('HU', rotation=90)
    
    img = ax[2].imshow(noiseS[:,:,101],cmap="gray")
    ax[2].set_axis_off()
    cbar = fig.colorbar(img, ax=ax[2])
    cbar.set_label('HU', rotation=90)
    
    img = ax[3].imshow(roi_final[:,:,101],cmap="gray")
    ax[3].set_axis_off()
    cbar = fig.colorbar(img, ax=ax[3])
    cbar.set_label('HU', rotation=90)
    
    plt.savefig(outDir+"simulated images.png")
    
    plt.close("all")

    # noise_rois = []
    # for ind in range(100):
    #     noise = noiseNormal(roi.shape)
    #     noise_filt = ndimage.gaussian_filter(noise,1)
    #     noise_rois.append(noise_filt)
    
    # NPS, freqs = computeNPS(noise_rois,voxelSize)
    # NPS0, freqs = computeNPS([noise_rois[0]],voxelSize)
    
    # plt.figure()    
    # plt.close("all")
    # plt.plot(freqs[0],NPS[:,0,0])
    # plt.plot(freqs[0],NPS0[:,0,0])
    
    # plt.plot(noise_rois[0][:,0,0])
    # plt.plot(noise_rois[1][:,0,0])
    
    
    