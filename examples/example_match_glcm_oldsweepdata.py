#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 10:06:10 2021

@author: qcao
"""

from __future__ import print_function

import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.phantoms.TrabeculaeVoronoi import *
import numpy as np
import matplotlib.pyplot as plt
# import mcubes
 
from bonebox.FEA.fea import *

import logging
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

import nrrd
import glob

# From: example_Voronoi_rods_FEA_phantomx_radiomics.py
def makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState):

    # Parameters for generating phantom mesh
    Sxyz, Nxyz = (5,5,5), (Nseeds, Nseeds, Nseeds) # volume extent in XYZ (mm), number of seeds along XYZ
    Rxyz = 1.
    # edgesRetainFraction = 0.5
    facesRetainFraction = 0.5
    # dilationRadius = 3 # (voxels)
    # randState = 123 # for repeatability
    
    # Parameters for generating phantom volume
    volumeSizeVoxels = (100,100,100)
    voxelSize = np.array(Sxyz) / np.array(volumeSizeVoxels)
    
    # Generate faces and edges
    points = makeSeedPointsCartesian(Sxyz, Nxyz)
    ppoints = perturbSeedPointsCartesianUniformXYZ(points, Rxyz, randState=randState)
    vor, ind = applyVoronoi(ppoints, Sxyz)
    uniqueEdges, uniqueFaces = findUniqueEdgesAndFaces(vor, ind)
    
    # Compute edge cosines
    edgeVertices = getEdgeVertices(vor.vertices, uniqueEdges)
    edgeCosines = computeEdgeCosine(edgeVertices, direction = (0,0,1))
    
    # Compute face properties
    faceVertices = getFaceVertices(vor.vertices, uniqueFaces)
    faceAreas = computeFaceAreas(faceVertices)
    faceCentroids = computeFaceCentroids(faceVertices)
    faceNormas = computeFaceNormals(faceVertices)
    
    # Filter random edges and faces
    uniqueEdgesRetain, edgesRetainInd = filterEdgesRandomUniform(uniqueEdges, 
                                                                 edgesRetainFraction, 
                                                                 randState=randState)
    uniqueFacesRetain, facesRetainInd = filterFacesRandomUniform(uniqueFaces, 
                                                                 facesRetainFraction, 
                                                                 randState=randState)
    
    volume = makeSkeletonVolumeEdges(vor.vertices, uniqueEdgesRetain, voxelSize, volumeSizeVoxels)
    volumeDilated = dilateVolumeSphereUniform(volume, dilationRadius)
    
    volumeDilated = volumeDilated.astype(float)
    bvtv = np.sum(volumeDilated>0) / volume.size
    
    edgeVerticesRetain = getEdgeVertices(vor.vertices, uniqueEdgesRetain)
    
    volumeDilated = setEdgesZero(volumeDilated)
    
    return volumeDilated, bvtv, edgeVerticesRetain

def getRadiomicFeatureNames():
    
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    settings['imageType'] = ['original','wavelet']
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)  
    
    # Extract radiomics from volume
    volume = np.random.rand(3,3,3)*256
    volumeSITK = sitk.GetImageFromArray(volume)
    maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
    featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    # featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    featureNames = list(featureVector.keys())
    
    return featureNames

def computeRadiomicFeatures(volume):
    
    # Define settings for signature calculation
    # These are currently set equal to the respective default values
    settings = {}
    settings['binWidth'] = 25
    settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
    settings['interpolator'] = sitk.sitkBSpline
    settings['imageType'] = ['original','wavelet']
    
    # Initialize feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)  
    
    # Extract radiomics from volume
    volumeSITK = sitk.GetImageFromArray(volume)
    maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
    featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
    featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
    
    return featureVectorArray

if __name__ == "__main__":
    
    plt.ion()
    
    rhoBone = 2e-3 # g/mm3
    voxelSize = (0.05, 0.05, 0.05) # mm
    pixelSize = (0.05, 0.05) # mm
    radiusTBS = 5 # pixels
    plattenThicknessVoxels = 5 # voxels
    plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm
    
    # Get radiomic features for GLCM
    save_dir = "/data/BoneBox-out/topopt/lazy_v3_sweep/"
    featuresROI = np.load(save_dir+"featuresROI.npy")
    
    # Test extract
    featureNames = getRadiomicFeatureNames()
    
    #%%
    
    def getFeaturesContaining(featureString):
        """
        Get Features from phanFeatures and featureNames Containing A Specific Substring
        
        phanFeatures
        featureNames
        isobvtv_xyeval
        
        """
        
        pf = phanFeatures[:,:,:,np.where([featureString in fn for fn in featureNames])[0]]
        fn = [fn for fn in featureNames if (featureString in fn)]
        
        bvtv = np.zeros(pf.shape)
        radius = np.zeros(pf.shape)
        Nseeds = np.zeros(pf.shape)
        
        for ii, isoval in enumerate(isobvtv_vals): # 3
            
            dilationRadiusArray = isobvtv_xyeval[ii][:,0]
            NseedsArray = isobvtv_xyeval[ii][:,1].astype(int)
            
            for pp in range(samplesPerVal): # 7
                
                for rr, randState in enumerate(randStates): # 14
                    
                    xind = ii*samplesPerVal+pp
                    yind = rr
                    
                    bvtv[xind, yind,:] = isoval
                    radius[xind,yind,:] = dilationRadiusArray[pp]
                    Nseeds[xind,yind,:] = NseedsArray[pp]
    
    return pf, fn, bvtv, radius, Nseeds
    
    #%%
    
    all_dilationRadius = np.linspace(1,3,10)
    all_Nseeds = np.arange(3,10).astype(int)
    all_edgesRetainFraction = 0.8
    all_randState = np.arange(3)
    
    featuresArr = np.zeros((len(all_dilationRadius),len(all_Nseeds),len(all_randState), 93))
    
    for rr, dilationRadius in enumerate(all_dilationRadius):
        for nn, Nseeds in enumerate(all_Nseeds):
            for rd, randState in enumerate(all_randState):
                
                edgesRetainFraction=0.8
                
                print(str(rr) +" "+ str(nn)+" "+str(rd))
        
                volumeDilated, bvtv, edgeVerticesRetain = makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState)
                featuresArr[rr,nn,rd,:] = computeRadiomicFeatures(volumeDilated)
                
                np.save(save_dir+"featuresArr",featuresArr)
                
    #%%
    
    