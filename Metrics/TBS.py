# -*- coding: utf-8 -*-

"""
Metrics.TBS

Trabecular Bone Score for Projection Images

Author:  Qian Cao

Based on:
    
    Pothuaud et al. Correlations between grey-level variations in 
    2D projection images (TBS) and 3D microarchitecture: 
    Applications in the study of human trabecular bone microarchitecture
    https://www.sciencedirect.com/science/article/pii/S8756328207008666

"""

import numpy as np

def computeImageTBS(image, radius=10, pixelSize=(1,1)):
    # image : 2D np.ndarray.
    # radius (pixels) : radius for computing pixel-wise TBS (radius=10 will result in 21**2-pixel ROI).
    # pixelSize (mm) : used to scale final TBS value.
    
    pixelSize = np.array(pixelSize)
    
    # Output TBS image
    # imageTBS = np.empty(np.array(image.shape)-2*radius)
    imageTBS = np.empty(np.array(image.shape))
    imageTBS[:] = np.nan
    
    # Coordinates in image where TBS is computed.
    indX = range(radius, image.shape[0]-radius)
    indY = range(radius, image.shape[1]-radius)
    
    # Coordinates of the ROI centered on pixel where TBS is evaluated.
    indR = np.arange(-radius, radius+1)
    xx, yy = np.meshgrid(indR*pixelSize[0], indR*pixelSize[1])
    k = np.sqrt(xx**2+yy**2).flatten()
    logk = np.log(k)
    
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

if __name__ == "__main__":
    
    import nrrd
    import matplotlib.pyplot as plt
    
    rhoBone = 2e-3 # g/mm3
    voxelSize = (0.05, 0.05, 0.05) # mm
    pixelSize = (0.05, 0.05) # mm
    radiusTBS = 4 # pixels
    
    # Simulate a simple projection image and convert units to aBMD
    roiBone, header = nrrd.read("../data/isodata_04216_roi_4.nrrd")
    roiBone[roiBone==255] = 1 # units for this is volume
    projectionImage = np.prod(np.array(voxelSize)) * rhoBone * np.sum(roiBone,axis=0).T \
         / np.prod(np.array(pixelSize))
    # [mm3][g/mm3]/[mm2]
    
    projectionTBS = computeTBS(projectionImage, radius=radiusTBS, pixelSize=pixelSize)
    
    plt.figure(figsize=(12.6, 4.66))
    plt.subplot(1,2,1)
    plt.imshow(projectionImage, cmap="gray")
    plt.axis("off")
    plt.colorbar()

    plt.subplot(1,2,2)
    plt.imshow(projectionTBS)
    plt.axis("off")
    plt.colorbar()