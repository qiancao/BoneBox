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

def getFeaturesContaining(featureString,featureNames,featureArr):
    
    return featureArr[:,:,:,np.where([featureString in fn for fn in featureNames])[0]]

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
    
    all_dilationRadius = np.linspace(1,5,20)
    all_Nseeds = np.arange(2,20).astype(int)
    all_edgesRetainFraction = 0.8
    all_randState = np.arange(5)
    
    featuresArr = np.zeros((len(all_dilationRadius),len(all_Nseeds),len(all_randState), 93))
    
    for rr, dilationRadius in enumerate(all_dilationRadius):
        for nn, Nseeds in enumerate(all_Nseeds):
            for rd, randState in enumerate(all_randState):
                
                edgesRetainFraction=0.8
                
                print(str(rr) +" "+ str(nn)+" "+str(rd))
        
                volumeDilated, bvtv, edgeVerticesRetain = makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState)
                featuresArr[rr,nn,rd,:] = computeRadiomicFeatures(volumeDilated*255)
                
                np.save(save_dir+"featuresArr",featuresArr)
    
    
    #%% Compares feature vectors
    
    featureTypeList = ["glcm", "gldm", "glrlm",'glszm','ngtdm']
    
    for featureType in featureTypeList:
    
        featuresArrSubset = getFeaturesContaining(featureType, featureNames, featuresArr)
        featuresROISubset = featuresROI[:,np.where([featureType in fn for fn in featureNames])[0]]
        
        distArr = np.zeros((len(all_dilationRadius),len(all_Nseeds),len(all_randState)))
        
        from scipy import spatial
        tree = spatial.KDTree(featuresROISubset)
        
        for rr, dilationRadius in enumerate(all_dilationRadius):
            for nn, Nseeds in enumerate(all_Nseeds):
                for rd, randState in enumerate(all_randState):
                    
                    queryFeatures = featuresArrSubset[rr,nn,rd,:]
                    ind = tree.query(queryFeatures)[1]
                    featuresClosestROI = featuresROISubset[ind,:]
                    
                    #%%
                    
                    # plt.figure()
                    # plt.plot(featuresROISubset.T,color = 'gray')
                    # plt.plot(queryFeatures,'bo-')
                    # plt.plot(featuresClosestROI,"rv-")
                    # plt.yscale("log")
                    # plt.xlabel("Features")
                    # plt.ylabel("Feature Values")                    
                    
                    #%%
                    
                    distArr[rr,nn,rd] = np.linalg.norm(queryFeatures-featuresClosestROI)
                    
        # Save feature vectors
        np.save(save_dir+"distArr_"+featureType, distArr)
        np.save(save_dir+"featuresArrSubset_"+featureType, featuresArrSubset)
                    
        plt.figure()
        plt.imshow(np.mean(np.log10(distArr),axis=2),interpolation="spline16")
        plt.title("Mean of Log NN Distance to Bone ROI "+ featureType +" Features")
        plt.xlabel("Dilation Radius (Thickness)")
        plt.ylabel("Nseeds (Structure Density)")
        plt.colorbar()
        plt.savefig(save_dir+"SweepFigure_meandist_"+featureType)
        plt.close("all")
        
        plt.figure()
        plt.imshow(np.std(np.log10(distArr),axis=2),interpolation="spline16",cmap="plasma")
        plt.title("Std of Log NN Distance to Bone ROI "+ featureType +" Features")
        plt.xlabel("Dilation Radius (Thickness)")
        plt.ylabel("Nseeds (Structure Density)")
        plt.colorbar()
        plt.savefig(save_dir+"SweepFigure_stddist_"+featureType)
        plt.close("all")
    
    