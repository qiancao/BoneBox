# -*- coding: utf-8 -*-
"""

Compute TBS for rois in ../data/rois

Created on Fri May 14 13:09:40 2021

@author: Qian.Cao

"""

import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.metrics import TBS
import glob
import nrrd
import numpy as np
import matplotlib.pyplot as plt

# Projection/TBS Settings
rhoBone = 2e-3 # g/mm3
voxelSize = (0.05, 0.05, 0.05) # mm
pixelSize = (0.05, 0.05) # mm
radiusTBS = 5 # pixels

out_dir = r"C://Users//Qian.Cao//tmp//"
roi_dir = '../data/rois/'

def readROI(filename):
    roiBone, header = nrrd.read(filename)
    roiBone[roiBone==255] = 1 # units for this is volume
    return roiBone
    
def computeROIProjection(roiBone, projectionAxis):
    projectionImage = np.prod(np.array(voxelSize)) * rhoBone * np.sum(roiBone,axis=projectionAxis).T \
         / np.prod(np.array(pixelSize))
    return projectionImage

def computeProjectionTBS(projectionImage):
    projectionTBS = TBS.computeTBSImage(projectionImage, radius=radiusTBS, pixelSize=pixelSize)
    return projectionTBS
    
def computeMeanTBS(projectionImage):
    projectionTBS = computeProjectionTBS(projectionImage)
    meanTBS = np.nanmean(projectionTBS)
    return meanTBS

Nrois  = len(glob.glob(roi_dir+"isodata_*_roi_*.nrrd"))

meanTBSList = []

for ind in range(Nrois):
    print(ind)
    fn = glob.glob(roi_dir+"isodata_*_roi_"+str(ind)+".nrrd")[0]
    roiBone = readROI(fn)
    for pind in range(2):
        projectionImage = computeROIProjection(roiBone, pind)
        meanTBS = computeMeanTBS(projectionImage)
        meanTBSList.append(meanTBS)
        print(meanTBS)
        
np.save(out_dir+"meanTBSList", np.array(meanTBSList))