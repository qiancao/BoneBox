# -*- coding: utf-8 -*-
"""
@author: Qian.Cao
"""

import nrrd
import numpy as np
import os, sys
from scipy import ndimage
import matplotlib.pyplot as plt
import os

import logging
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
import nrrd
import glob
import umap

from sklearn.model_selection import train_test_split

import sys
sys.path.append("../bonebox/metrics/")

from FeaturesRadiomics import *

roiDir = "../data/rois/"

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
         Froi = Froi + np.abs(np.fft.fftn(roi, axes=tuple(range(ndim))))**2
    
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

def simulateImage(roi, voxSize, voxSizeNew, stdMTF, noise_std, seed=None):
    
    roi = applySampling(roi, voxSize, voxSizeNew)
    voxSize = voxSizeNew
    
    # # normalize to 0 and 1, scale to bone HU (this is done outside the function now via applyScaling)
    # roi = (roi - np.min(roi)) / (np.max(roi) - np.min(roi))
    # roi = roi * boneHU
    
    # get frequency axis
    freqs = getFreqs(roi.shape,voxSizeNew)
    
    T = make3DMTFGaussian(freqs,stdMTF)
    roiT = applyMTF(roi,T)
    
    ramp = makeNPSRamp(freqs)
    
    S = (T**2)*ramp
    S = S / integrateNPS(S,freqs) * noise_std # TODO: should be squqred
    
    noiseS = NPS2noise(S,seed=seed) * (noise_std**2)
    
    roi_final = roiT+noiseS
    
    return roi_final, noiseS, roiT, S, T

def applyScaling(roi):
    roi = (roi - np.min(roi)) / (np.max(roi) - np.min(roi))
    roi = roi * boneHU
    return roi

def getROI(number):
    filenameNRRD = glob.glob(roiDir+f"*_roi_{number}.nrrd")[0]
    roi, header = nrrd.read(filenameNRRD)
    roi = applyScaling(roi)
    return roi

def pathBetsy(path):
    return path.replace("/gpfs_projects/qian.cao/BoneBox-out","/projects01/didsr-aiml/qian.cao/BoneBox-out")

if __name__ == "__main__":
    
    plt.close("all")

    # features directory
    featureDir = pathBetsy("/gpfs_projects/qian.cao/BoneBox-out/test_20220422_bin_cross_parallel/")
    os.makedirs(featureDir,exist_ok=True)

    features = np.load(featureDir+"featuresArray.npy")
    
    # models directory
    modelDir = pathBetsy("/gpfs_projects/qian.cao/BoneBox-out/test_20220422_bin_cross_parallel_biomarker/")
    os.makedirs(modelDir,exist_ok=True)
    
    importances = np.load(os.path.join(modelDir,"importances.npy"))
    y_test = np.load(os.path.join(modelDir,"y_test.npy"))
    y_preds = np.load(os.path.join(modelDir,"y_preds.npy"))

    # output directory
    outDir = pathBetsy("/gpfs_projects/qian.cao/BoneBox-out/test_20220422_bin_cross_parallel_figures_SPIE/")
    os.makedirs(outDir,exist_ok=True)
    
    # load stiffness data: see example_rois... 20211116
    fem_dir = "../data/"
    roi_delta_z = np.load(fem_dir+"roi_delta_z.npy")
    roi_num_nodes = np.load(fem_dir+"roi_num_nodes.npy")
    roi_vm_mean = np.load(fem_dir+"roi_vm_mean.npy") # This is mean stress.
    roi_bvtv = np.load(fem_dir+"roi_bvtv.npy") # This is BVTV
    
    # feature names
    featureNames = getRadiomicFeatureNames()
    
    # image parameters
    voxSize = (0.05, 0.05, 0.05) # mm
    boneHU = 1800

    # parameter space
    nScales = np.linspace(1.2, 0.2, 60) # change noise only # sweeps across noise and resolution settings
    rScales = np.linspace(1, 0.3, 40)
    vscale = 1

    # Size of the test split
    num_bones_test = 7 # number of bones reserved for testing
    test_split_size = num_bones_test/16
    num_test = int(num_bones_test*13)

    feat = features[:,:,0,0,0]
    _, _, bvtv_train, bvtv_test = train_test_split(feat, roi_bvtv, test_size=test_split_size, random_state = 3, shuffle=False)

    
    
    




    # vscale = 1
    
    # print(f"noise scaling: {nScales}")
    # print(f"resolution scaling: {rScales}")
    # print(f"voxel scaling: {vscale}")
    
    # #%%
    
    # ROI_VISUALIZE = [4,10]
    
    # numROIs = 208
    # seeds = np.array(range(5))
    
    # featuresArray = np.zeros((numROIs, len(featureNames), nScales.size, rScales.size, len(seeds)))
    
    # for indNoise, nscale in enumerate(nScales):
    #     for indResolution, rscale in enumerate(rScales):
        
    #         print(f"Imaging Config: {indNoise} {indResolution}")
            
    #         voxSizeNew = np.array((0.156,0.156,0.2)) * vscale  # mm
    #         noise_std = 180 * nScales[indNoise]
    #         stdMTF = np.array([1.8,1.8,0.5]) * rScales[indResolution]
            
    #         for s, seed in enumerate(seeds):
                
    #             print(f"Seed: {seed}")
                
    #             plt.ioff()
                
    #             # simulate images
    #             roi_imgs = [] # list of simulated images
    #             for rind in range(numROIs):
    #                 print(f"simulating roi: {rind}")
    #                 roi = getROI(rind)
    #                 roi_img, noiseS, roiT, S, T = simulateImage(roi, voxSize, voxSizeNew, stdMTF, noise_std, seed)
    #                 roi_imgs.append(roi_img)
                    
    #                 if rind in ROI_VISUALIZE:
                        
    #                     plt.figure()
    #                     plt.imshow(roi_img[:,:,roi_img.shape[2]//2], cmap="gray", vmin=700-3800/2,vmax=700+3800/2)
    #                     plt.title(f"ROI: {rind}, Sim: {indNoise} {indResolution}")
    #                     plt.savefig(outDir+f"img_{rind}_{indNoise}_{indResolution}.png")
    #                     plt.close("all")
                        
    #                     if indNoise == 0 and indResolution == 0:
                            
    #                         plt.figure(visible=False)
    #                         plt.imshow(roi[:,:,roi.shape[2]//2], cmap="gray", vmin=700-3800/2,vmax=700+3800/2)
    #                         plt.title(f"ROI: {rind}")
    #                         plt.savefig(outDir+f"img_{rind}.png")
    #                         plt.close("all")
                    
    #             print(f"len(roi_imgs) {len(roi_imgs)}")
                    
    #             print(f"computing features")
    #             featureNamesList, featureVectorArray = computeRadiomicFeaturesParallel(roi_imgs, numWorkers=20)
    #             featuresArray[:,:,indNoise,indResolution,s] = featureVectorArray
    #             print(f"computing features: finished")
    
    #         np.save(outDir+"featuresArray",featuresArray)
        
    #%%
    
    # corrArray = np.zeros((len(featureNames),vScales.size))
    # stdSampleArray = np.zeros((len(featureNames),vScales.size,len(seeds))) # standard deviation along samples
    # stdRepeatArray = np.zeros((len(featureNames),vScales.size,numROIs)) # standard deviation along noise realizations
    # meanArray = np.zeros((len(featureNames),vScales.size))
    
    # for ind, vscale in enumerate(vScales):
    #     for find, feature in enumerate(featureNames):
        
    #         corrArray[find,ind] = np.corrcoef(featuresArray[:,find,ind,1],roi_vm_mean[:numROIs])[0,1]**2
            
    # stdSampleArray = np.std(featuresArray,axis=0)
    # meanSampleArray = np.mean(featuresArray,axis=0)
    # cvSampleArray =  stdSampleArray / np.abs(meanSampleArray)
    
    # stdRepeatArray = np.std(featuresArray,axis=3)
    # meanRepeatArray = np.mean(featuresArray,axis=3)
    # cvRepeatArray = stdRepeatArray / np.abs(meanRepeatArray)
    
    # stdArray = np.std(featuresArray,axis=(0,3))
    # meanArray = np.mean(featuresArray,axis=(0,3))
    # cvArray = stdArray / np.abs(meanArray)
    
    # #%% Overall R2
    
    # plt.close("all")
    
    # plt.figure(figsize=(6,15))
    # plt.imshow(corrArray,interpolation='nearest', aspect='auto')
    # plt.xticks([])
    # plt.yticks(np.arange(len(featureNames)),labels=featureNames,fontsize=8)
    # plt.colorbar(orientation="horizontal",pad=0.01)
    # plt.tight_layout()
    
    # plt.savefig(outDir+"r2.png")
    
    # #%% Variation with Respect to Total STD
    
    # plt.close("all")
    
    # plt.figure(figsize=(6,15))
    # plt.imshow(cvArray,interpolation='nearest', aspect='auto',cmap="inferno_r")
    # plt.xticks([])
    # plt.yticks(np.arange(len(featureNames)),labels=featureNames,fontsize=8)
    # plt.colorbar(orientation="horizontal",pad=0.01)
    # plt.tight_layout()
    
    # plt.savefig(outDir+"cv.png")
    
    # #%% Variation with Respect to Noise
    
    # plt.close("all")
    
    # plt.figure(figsize=(6,15))
    # plt.imshow(np.mean(cvRepeatArray,axis=0),interpolation='nearest', aspect='auto',cmap="inferno_r")
    # plt.xticks([])
    # plt.yticks(np.arange(len(featureNames)),labels=featureNames,fontsize=8)
    # plt.colorbar(orientation="horizontal",pad=0.01)
    # plt.tight_layout()
    
    # plt.savefig(outDir+"cvRepeatArray.png")
    
    # #%% Portion of 
    
    # plt.close("all")
    
    # plt.figure(figsize=(6,15))
    # plt.imshow(np.mean(cvRepeatArray,axis=0)/cvArray,interpolation='nearest', aspect='auto',cmap="cividis_r")
    # plt.xticks([])
    # plt.yticks(np.arange(len(featureNames)),labels=featureNames,fontsize=8)
    # plt.colorbar(orientation="horizontal",pad=0.01)
    # plt.tight_layout()
    
    # plt.savefig(outDir+"cv_fraction.png")
    
    # #%%
    
    # stdSampleArray = np.std(featuresArray,axis=0)
    
    
    # plt.figure()
    # plt.imshow(np.ones(2,2))
    # plt.colorbar(orientation="horizontal")
    
    # #%% Preview MTF and NPS
    
    # sh = roi.shape
    
    # # coneshift = np.fft.fftshift(cone)
    # Tshift = np.fft.fftshift(T)
    # Sshift = np.fft.fftshift(S)
    # freqx = np.fft.fftshift(freqs[0])
    
    # roi_final = roiT+noiseS
    
    # axlims = [np.min(freqs[0]), np.max(freqs[0]), 
    #           np.min(freqs[1]), np.max(freqs[1]),
    #           np.min(freqs[2]), np.max(freqs[2])]
    
    # fig, ax = plt.subplots(2,2,figsize=(12,10))
    # img = ax[0,0].imshow(Tshift[:,:,sh[2]//2],cmap="viridis",extent=axlims[0:4])
    # ax[0,0].set_xlabel("X (1/mm)")
    # ax[0,0].set_ylabel("Y (1/mm)")
    # cbar = fig.colorbar(img, ax=ax[0,0])
 
    # img = ax[1,0].imshow(Tshift[:,sh[1]//2,:].T,cmap="viridis",extent=[np.min(freqs[0]), np.max(freqs[0]),np.min(freqs[2]), np.max(freqs[2])])
    # ax[1,0].set_xlabel("X (1/mm)")
    # ax[1,0].set_ylabel("Z (1/mm)")
    # cbar = fig.colorbar(img, ax=ax[1,0])
    
    # img = ax[0,1].imshow(Sshift[:,:,sh[2]//2],cmap="gnuplot2",extent=axlims[0:4])
    # ax[0,1].set_xlabel("X (1/mm)")
    # ax[0,1].set_ylabel("Y (1/mm)")
    # cbar = fig.colorbar(img, ax=ax[0,1])
    # cbar.set_label('HU^2 mm^3', rotation=90)
    
    # img = ax[1,1].imshow(Sshift[:,sh[1]//2,:].T,cmap="gnuplot2",extent=[np.min(freqs[0]), np.max(freqs[0]),np.min(freqs[2]), np.max(freqs[2])])
    # ax[1,1].set_xlabel("X (1/mm)")
    # ax[1,1].set_ylabel("Z (1/mm)")
    # cbar = fig.colorbar(img, ax=ax[1,1])
    # cbar.set_label('HU^2 mm^3', rotation=90)
    
    # plt.savefig(outDir+"3D Metrics.png")
    
    # #%% 
    
    # plt.close("all")
    
    # fig, ax = plt.subplots(1,2,figsize=(12,4))
    
    # ax[0].plot(freqx,Tshift[:,sh[1]//2,sh[2]//2],'k')
    # ax[0].set_xlabel("Freqency (1/mm)")
    # ax[0].set_xlim([0,np.max(freqx)])
    # ax[0].set_title("In-plane MTF")
    
    # ax[1].plot(freqx,Sshift[:,sh[1]//2,sh[2]//2],'k')
    # ax[1].set_xlabel("Freqency (1/mm)")
    # ax[1].set_ylabel("(HU^2 mm^3)")
    # ax[1].set_xlim([0,np.max(freqx)])
    # ax[1].set_title("In-plane NPS")
    
    # plt.savefig(outDir+"3D Metrics in-plane.png")
    
    # #%% Image Preview
    # fig, ax = plt.subplots(2,2,figsize=(12,10))
    
    # ax = ax.flatten()
    # img = ax[0].imshow(roi0[:,:,roi0.shape[2]//2],cmap="gray",vmin=700-3800/2,vmax=700+3800/2)
    # ax[0].set_axis_off()
    # cbar = fig.colorbar(img, ax=ax[0])
    # cbar.set_label('HU', rotation=90)
    
    # img = ax[1].imshow(roiT[:,:,sh[2]//2],cmap="gray",vmin=700-3800/2,vmax=700+3800/2)
    # ax[1].set_axis_off()
    # cbar = fig.colorbar(img, ax=ax[1])
    # cbar.set_label('HU', rotation=90)
    
    # img = ax[2].imshow(noiseS[:,:,sh[2]//2],cmap="gray",vmin=700-3800/2,vmax=700+3800/2)
    # ax[2].set_axis_off()
    # cbar = fig.colorbar(img, ax=ax[2])
    # cbar.set_label('HU', rotation=90)
    
    # img = ax[3].imshow(roi_final[:,:,sh[2]//2],cmap="gray",vmin=700-3800/2,vmax=700+3800/2)
    # ax[3].set_axis_off()
    # cbar = fig.colorbar(img, ax=ax[3])
    # cbar.set_label('HU', rotation=90)
    
    # plt.savefig(outDir+"simulated images.png")
    
  