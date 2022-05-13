#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:04:55 2022

@author: qian.cao
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import nrrd

dataDir = "/gpfs_projects/qian.cao/BoneBox/data/rois/"
outDir = "/gpfs_projects/qian.cao/BoneBox/data/rois/previews/"
outROIs = "/gpfs_projects/qian.cao/BoneBox/data/roi_indices_trabecular"
os.makedirs(outDir,exist_ok=True)

dataFiles = glob.glob(dataDir+"*.nrrd")
numROIs = 208

plt.ioff()

for ind in range(numROIs):
    
    print(f"{ind}")
    
    imgName = glob.glob(dataDir+f"*_roi_{ind}.nrrd")[0]
    img, header = nrrd.read(imgName)
    
    fig, axs = plt.subplots(1,3,figsize=(12,6))
    axs[0].imshow(img[100,:,:].T,cmap="gray",interpolation="nearest");axs[0].axis("off");
    axs[1].imshow(img[:,100,:].T,cmap="gray",interpolation="nearest");axs[1].axis("off");
    axs[2].imshow(img[:,:,100].T,cmap="gray",interpolation="nearest");axs[2].axis("off");
    plt.savefig(outDir+f"roi_{ind}.png")
    plt.close("all")

# exclude ROIs where volume is mostly empty space dominated by cortical bone
excludeROIs = [13,14,15,26,27,28,39,40,41,65,67,78,80,91,93,104,106,110,111,112,132,145,156,157,158,189,195,197]
trabecularROIs = list(set(range(208)) - set(excludeROIs))
trabecularROIs.sort()
np.save(outROIs,np.array(trabecularROIs).sort())