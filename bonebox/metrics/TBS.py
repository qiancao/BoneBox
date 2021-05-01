# -*- coding: utf-8 -*-

"""

metrics.TBS

Trabecular Bone Score for Projection Images

Author:  Qian Cao

Based on:
    
    Pothuaud et al. Correlations between grey-level variations in 
    2D projection images (TBS) and 3D microarchitecture: 
    Applications in the study of human trabecular bone microarchitecture
    https://www.sciencedirect.com/science/article/pii/S8756328207008666

"""

import numpy as np

def getIndXY(imageShape, radius=10):
    # Coordinates in image where TBS is computed.
    
    indX = range(radius, imageShape[0]-radius)
    indY = range(radius, imageShape[1]-radius)
    
    return indX, indY

def getLogk(radius=10, pixelSize=(1,1)):
    # Coordinates of the ROI centered on pixel where TBS is evaluated.
    
    indR = np.arange(-radius, radius+1)
    xx, yy = np.meshgrid(indR*pixelSize[0], indR*pixelSize[1])
    k = np.sqrt(xx**2+yy**2).flatten()
    logk = np.log(k)
    
    return logk

def computeTBSImage(image, radius=10, pixelSize=(1,1)):
    # image : 2D np.ndarray.
    # radius (pixels) : radius for computing pixel-wise TBS (radius=10 will result in 21**2-pixel ROI).
    # pixelSize (mm) : used to scale final TBS value.
    
    pixelSize = np.array(pixelSize)
    
    # Output TBS image
    # imageTBS = np.empty(np.array(image.shape)-2*radius)
    imageTBS = np.empty(np.array(image.shape))
    imageTBS[:] = np.nan
    
    # Set up ROI
    indX, indY = getIndXY(image.shape, radius)
    logk = getLogk(radius, pixelSize)
    
    # Mask out center pixel: ind=len(logk)//2
    mask = np.ones(logk.shape,dtype=bool)
    mask[len(logk)//2] = 0
    
    for x in indX:
        for y in indY:
            V = (image[x-radius:x+radius+1, y-radius:y+radius+1].flatten() - image[x,y])**2
            logV = np.log(V)
            idx = mask & np.isfinite(logV)
            linfit = np.polyfit(logk[idx],logV[idx],1)
            imageTBS[x,y] = linfit[0]
    
    return imageTBS

def updateMeanStdN(mean0, std0, n0, x, idx):
    # Welford's running standard deviation algorithm
    # based on https://www.kite.com/python/answers/how-to-find-a-running-standard-deviation-in-python
    
    mean1, std1, n1 = mean0, std0, n0
    
    n1[idx] = n0[idx] + 1
    mean1[idx] = mean0[idx] + (x[idx]-mean0[idx]) / n0[idx]
    std1[idx] = std0[idx] + (x[idx] - mean0[idx]) * (x[idx] - mean1[idx])
    
    return mean1, std1, n1

def computeTBSVariogram(image, radius=10, pixelSize=(1,1)):
    # computes log variogram (running mean and std)
    
    # Set up ROI
    indX, indY = getIndXY(image.shape, radius)
    logk = getLogk(radius, pixelSize)
    
    # Mean and standard deviation of logVs
    logVmean = np.zeros(logk.shape)
    logVstd = np.zeros(logk.shape)
    logVn = np.zeros(logk.shape) # keeps track of number of valid points
    
    # Mask out center pixel: ind=len(logk)//2
    mask = np.ones(logk.shape,dtype=bool)
    mask[len(logk)//2] = 0
    
    for x in indX:
        for y in indY:
            V = (image[x-radius:x+radius+1, y-radius:y+radius+1].flatten() - image[x,y])**2
            logV = np.log(V)
            idx = mask & np.isfinite(logV)
            
            # Tabulate running mean and standard deviation
            logVmean, logVstd, logVn = updateMeanStdN(logVmean, logVstd, logVn, logV, idx)
    
    return logk, logVmean, logVstd, logVn

if __name__ == "__main__":
    
    import nrrd
    import matplotlib.pyplot as plt
    
    rhoBone = 2e-3 # g/mm3
    voxelSize = (0.05, 0.05, 0.05) # mm
    pixelSize = (0.05, 0.05) # mm
    radiusTBS = 5 # pixels
    
    # Simulate a simple projection image and convert units to aBMD
    roiBone, header = nrrd.read("../../data/isodata_04216_roi_4.nrrd")
    roiBone[roiBone==255] = 1 # units for this is volume
    projectionImage = np.prod(np.array(voxelSize)) * rhoBone * np.sum(roiBone,axis=0).T \
         / np.prod(np.array(pixelSize))
    # [mm3][g/mm3]/[mm2]
    
    projectionTBS = computeTBSImage(projectionImage, radius=radiusTBS, pixelSize=pixelSize)
    logk, logVmean, logVstd, logVn = \
        computeTBSVariogram(projectionImage, radius=radiusTBS, pixelSize=pixelSize)
    
    plt.figure(figsize=(12.6, 4.66))
    plt.subplot(1,2,1)
    plt.imshow(projectionImage, cmap="gray")
    plt.axis("off")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(projectionTBS)
    plt.axis("off")
    plt.colorbar()