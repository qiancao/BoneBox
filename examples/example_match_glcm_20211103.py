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

import umap

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
    
    flag_rundata = False
    flag_normfeatures = True
    
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
    
    featureTypeList = ["glcm", "gldm", "glrlm",'glszm','ngtdm']
    
    if flag_rundata == True:
    
        featuresArr = np.zeros((len(all_dilationRadius),len(all_Nseeds),len(all_randState), 93))
        
        for rr, dilationRadius in enumerate(all_dilationRadius):
            for nn, Nseeds in enumerate(all_Nseeds):
                for rd, randState in enumerate(all_randState):
                    
                    edgesRetainFraction=0.8
                    
                    print(str(rr) +" "+ str(nn)+" "+str(rd))
            
                    volumeDilated, bvtv, edgeVerticesRetain = makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState)
                    featuresArr[rr,nn,rd,:] = computeRadiomicFeatures(volumeDilated*255)
                    
                    np.save(save_dir+"featuresArr",featuresArr)
    
    else:
        
        featuresArr = np.load(save_dir+"featuresArr.npy")

    # Normalize feature array to mean and standard deviation of the ROIs
    if flag_normfeatures:
        
        featuresROIMean = np.mean(featuresROI, axis=(0))
        featuresROIStd = np.std(featuresROI, axis=(0))
        
        featuresArr = (featuresArr-featuresROIMean[None,None,None,:])/featuresROIStd[None,None,None,:]
        featuresROI = (featuresROI-featuresROIMean[None,:])/featuresROIStd[None,:]

    else:

        def featuresNorm(f):
            return f
        
        # for completeness: unnormalized features
        featuresArr = featuresArr
        featuresROI = featuresROI
        
    featuresROI[np.isnan(featuresROI)] = 0
    featuresArr[np.isnan(featuresArr)] = 0
    
    featuresROI[np.isinf(featuresROI)] = 0
    featuresArr[np.isinf(featuresArr)] = 0
        
    #%% Compares feature vectors

    for featureType in featureTypeList:
    
        featuresArrSubset = getFeaturesContaining(featureType, featureNames, featuresArr)
        featuresROISubset = featuresROI[:,np.where([featureType in fn for fn in featureNames])[0]]
        
        if flag_normfeatures:
            featuresROISubsetMean = featuresROIMean[np.where([featureType in fn for fn in featureNames])[0]]
        
        distArr = np.zeros((len(all_dilationRadius),len(all_Nseeds),len(all_randState)))
        
        from scipy import spatial
        tree = spatial.KDTree(featuresROISubset)
        
        for rr, dilationRadius in enumerate(all_dilationRadius):
            for nn, Nseeds in enumerate(all_Nseeds):
                for rd, randState in enumerate(all_randState):
                    
                    # Nearest Neighbor
                    queryFeatures = featuresArrSubset[rr,nn,rd,:]
                    ind = tree.query(queryFeatures)[1]
                    featuresClosestROI = featuresROISubset[ind,:]
                    
                    if flag_normfeatures:
                        distArr[rr,nn,rd] = np.linalg.norm((queryFeatures-featuresClosestROI)/featuresROISubsetMean)
                    else:
                        distArr[rr,nn,rd] = np.linalg.norm(queryFeatures-featuresClosestROI)
                    
        # Save feature vectors
        np.save(save_dir+"distArr_"+featureType, distArr)
        np.save(save_dir+"featuresArrSubset_"+featureType, featuresArrSubset)
        
        # Output figures
        plt.figure()
        plt.imshow(np.mean(np.log10(distArr),axis=2),interpolation="none")
        plt.title("Mean of Log NN Distance to Bone ROI "+ featureType +" Features")
        plt.xlabel("Dilation Radius (Thickness)")
        plt.ylabel("Nseeds (Structure Density)")
        plt.colorbar()
        plt.savefig(save_dir+"SweepFigure_meandist_"+featureType)
        plt.close("all")
        
        plt.figure()
        plt.imshow(np.std(np.log10(distArr),axis=2),interpolation="none",cmap="plasma")
        plt.title("Std of Log NN Distance to Bone ROI "+ featureType +" Features")
        plt.xlabel("Dilation Radius (Thickness)")
        plt.ylabel("Nseeds (Structure Density)")
        plt.colorbar()
        plt.savefig(save_dir+"SweepFigure_stddist_"+featureType)
        plt.close("all")
        
    #%% UMAP feature analysis
    
    # directory for UMAP parameter tuning
    umap_dir = save_dir+"umap/"

    all_n_neighbors = [2,3,4,5,6,7,8,9,10,20,50,100, 150, 200]
    
    # First order mean for bone ROIs
    BMD = featuresROI[:,8]
    BMDLD = featuresLD[:,8]
    
    # Geometric phantom features
    featuresArrReshape = np.reshape(featuresArr,(-1,93))
    
    # Load LD bone phantom features and apply UMAP
    featuresLD = np.load(save_dir+"featuresReshaped.npy")
    if flag_normfeatures:
        featuresLD = (featuresLD-featuresROIMean[None,:])/featuresROIStd[None,:]
    featuresLD[np.isnan(featuresLD)] = 0
    featuresLD[np.isinf(featuresLD)] = 0
    
    #%% Sweep UMAP n_neighbors for ROI data
    for nn, n_neighbors in enumerate(all_n_neighbors):
        
        trans = umap.UMAP(n_neighbors=n_neighbors, random_state=42).fit(featuresROI)
        
        plt.figure(figsize=(10,10))
        plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s = 15, c = BMD, cmap='cool', vmin=-5, vmax=5)
        # plt.scatter(embeddingArr[:, 0], embeddingArr[:, 1], s= 5, c='r', cmap='Spectral')
        plt.title('UMAP Embedding for Radiomic Features', fontsize=16);
        plt.xlabel("Projection 0", fontsize=16)
        plt.ylabel("Projection 1", fontsize=16)
        plt.colorbar()
        plt.savefig(umap_dir+"UMAP ROI Nneighbors="+str(n_neighbors)+".png")
        plt.close("all")
        
        del trans
        
    #%% Sweep UMAP n_neighbors for ROI data
    
    for nn, n_neighbors in enumerate(all_n_neighbors):
        
        trans = umap.UMAP(n_neighbors=n_neighbors, random_state=42).fit(featuresLD)
        
        plt.figure(figsize=(10,10))
        plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s = 15, c = BMDLD, cmap='cool', vmin=-5, vmax=5)
        # plt.scatter(embeddingArr[:, 0], embeddingArr[:, 1], s= 5, c='r', cmap='Spectral')
        plt.title('UMAP Embedding for Radiomic Features', fontsize=16);
        plt.xlabel("Projection 0", fontsize=16)
        plt.ylabel("Projection 1", fontsize=16)
        plt.colorbar()
        plt.savefig(umap_dir+"UMAP LD Nneighbors="+str(n_neighbors)+".png")
        plt.close("all")
        
        del trans
    
    #%%
    
    # Fit UMAP to ROI data
    trans = umap.UMAP(n_neighbors=n_neighbors, random_state=42).fit(featuresROI)

    # Embedding
    embeddingArr = trans.transform(featuresArrReshape)
    embeddingLD = trans.transform(featuresLD)

    # Plot UMAP
    BMD = featuresROI[:,8]
    plt.figure(figsize=(10,10))
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s = 15, c = BMD, cmap='cool', vmin=-5, vmax=5)
    # plt.scatter(embeddingArr[:, 0], embeddingArr[:, 1], s= 5, c='r', cmap='Spectral')
    plt.title('UMAP Embedding for Radiomic Features', fontsize=16)
    plt.xlabel("Projection 0", fontsize=16)
    plt.ylabel("Projection 1", fontsize=16)
    plt.colorbar()
    plt.savefig(save_dir+"UMAP ROI.png")
    plt.close("all")
    
    BMDArr = featuresArrReshape[:,8]
    plt.figure(figsize=(10,10))
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s = 15, c='k', cmap='Spectral')
    plt.scatter(embeddingArr[:, 0], embeddingArr[:, 1], s= 5, c = BMDArr, cmap='cool', vmin=-5, vmax=5)
    plt.title('UMAP Embedding for Radiomic Features', fontsize=16);
    plt.xlabel("Projection 0", fontsize=16)
    plt.ylabel("Projection 1", fontsize=16)
    plt.colorbar()
    plt.savefig(save_dir+"UMAP ROI with Phantom.png")
    plt.close("all")
    
    BMDLD = featuresLD[:,8]
    plt.figure(figsize=(10,10))
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s = 15, c='k', cmap='Spectral')
    plt.scatter(embeddingLD[:, 0], embeddingLD[:, 1], s= 5, c = BMDLD, cmap='cool', vmin=-5, vmax=5)
    plt.title('UMAP Embedding for Radiomic Features', fontsize=16);
    plt.xlabel("Projection 0", fontsize=16)
    plt.ylabel("Projection 1", fontsize=16)
    plt.colorbar()
    plt.savefig(save_dir+"UMAP ROI with LD Phantoms.png")
    plt.close("all")
    
    