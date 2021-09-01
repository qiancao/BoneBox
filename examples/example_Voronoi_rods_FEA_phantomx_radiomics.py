# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:55:11 2021

@author: Qian.Cao

Generate a series of Voronoi Rod (AND PLATE) phantoms

"""

from __future__ import print_function

import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.phantoms.TrabeculaeVoronoi import *
import numpy as np
import matplotlib.pyplot as plt
# import mcubes
 
from bonebox.FEA.fea import *

plt.ion()

print('Running example for TrabeculaeVoronoi')

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

rhoBone = 2e-3 # g/mm3
voxelSize = (0.05, 0.05, 0.05) # mm
pixelSize = (0.05, 0.05) # mm
radiusTBS = 5 # pixels
plattenThicknessVoxels = 5 # voxels
plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm

def computeFEA(volume):

    roiBone = addPlatten(volume, plattenThicknessVoxels)
    vertices, faces, normals, values = Voxel2SurfMesh(roiBone, voxelSize=(0.05,0.05,0.05), step_size=1)
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    nodes, elements, tet = Surf2TetMesh(vertices, faces, verbose=0)
    feaResult = computeFEACompressLinear(nodes, elements, plattenThicknessMM)
    elasticModulus = computeFEAElasticModulus(feaResult)
    
    return elasticModulus

#% Generate Phantoms with different dilation Radii and seeds

out_dir = "/data/BoneBox-out/"
phantoms_dir = "/data/BoneBox-out/phantoms/"

# Projection/TBS Settings
all_randState = [1,2,3,4]
all_dilationRadius = np.linspace(1,3,10)
all_Nseeds = np.arange(3,10).astype(int)
edgesRetainFraction = 0.8

bvtvs = np.zeros((len(all_randState),len(all_Nseeds),len(all_dilationRadius)))
# Es = np.zeros(bvtvs.shape)

#% Load Existing BvTv

bvtvs = np.load(out_dir+"FEA_bvtvs_2.npy")

import scipy.ndimage

# plot mean and Std of bvtvs
meanbvtv = scipy.ndimage.zoom(np.mean(bvtvs,axis=0), 10, order=1)
extent=[np.min(all_dilationRadius),np.max(all_dilationRadius),
        np.min(all_Nseeds),np.max(all_Nseeds)]

fig, ax = plt.subplots()
ax.imshow(meanbvtv, interpolation="nearest",extent=extent,aspect='auto', origin='lower')
cs = ax.contour(meanbvtv, extent=extent, colors='w', origin='lower')
ax.clabel(cs, fontsize=9, inline=True)

plt.xlabel("Dilation Radius")
plt.ylabel("N")

# Get contour lines
samplesPerVal = 7
isobvtv_vals = [0.24, 0.32, 0.4]
isobvtv_xy = []
isobvtv_xyeval = []

for vv, isobvtv in enumerate(isobvtv_vals):
    cs = ax.contour(meanbvtv, [isobvtv], extent=extent)
    p = cs.collections[0].get_paths()[0]
    isobvtv_xy.append(p.vertices)
    
for ii, xy in enumerate(isobvtv_xy):
    
    xylist = []
    
    Nxy = xy.shape[0]
    Dxy = Nxy // (samplesPerVal-1)
    
    # select 4 evenly spaced points along the isocline
    for ss in range(samplesPerVal-1):
        xylist.append(xy[ss*Dxy,:])
    xylist.append(xy[-1,:])
    
    isobvtv_xyeval.append(np.array(xylist))
    
#% Phantom HU

# Parameters for PhantomX
mmPerInch = 25.4

HUBone = 1500
HUAir = -800
HUMarrow = -50

dimXcm = 27
dimYcm = 19

DPIxy = 300
voxXY = 1/(DPIxy/mmPerInch)
voxZ = 0.071428571 # mm

dimX = int(270//voxXY)
dimY = int(190//voxXY)
dimZ = 100

phantomsInX = dimX // 150 # 21 {3 BvTv} x {7 Samples}
phantomsInY = dimY // 150 # 14 RandStates

phanHU = np.ones((dimX,dimY,dimZ))*(-800)

randStates = np.arange(14)

#% Skip: Generate Phantoms

# for ii, isoval in enumerate(isobvtv_vals): # 3
    
#     dilationRadiusArray = isobvtv_xyeval[ii][:,0]
#     NseedsArray = isobvtv_xyeval[ii][:,1].astype(int)
    
#     for pp in range(samplesPerVal): # 7
        
#         for rr, randState in enumerate(randStates): # 14
            
#             print(ii, pp, rr)
            
#             xind = (ii*7+pp)*(dimX // phantomsInX)
#             yind = rr*(dimY // phantomsInY)
            
#             volume, bvtv, edgeVerticesRetain = makePhantom(dilationRadiusArray[pp],
#                                                             NseedsArray[pp],
#                                                             edgesRetainFraction,
#                                                             randState)
            
#             # Convert to HU
#             volume[volume==1] = HUBone
#             volume[volume==0] = HUMarrow
            
#             # Assign to phanHU
#             phanHU[xind:xind+100,yind:yind+100,:] = volume
            
# np.save(out_dir+"PhantomX_20210623", phanHU)

# #%

# import matplotlib.pyplot as plt
# import numpy as np
# plt.imshow(phanHU[:,:,50].T,cmap="gray")
# plt.axis("off")
# plt.savefig(out_dir+'PhantomX_20210623.png',bbox_inches = 'tight',
#     pad_inches = 0)

#% Shuffle - Load File and Compute Radiomics

phanHU = np.load(out_dir+"PhantomX_20210623.npy")
            
#%% Radiomics Extraction Pipeline to be incorporated into BoneBox.metrics.radiomics.py

import logging
import SimpleITK as sitk

import radiomics
from radiomics import featureextractor

# Define settings for signature calculation
# These are currently set equal to the respective default values
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkBSpline

# Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableFeatureClassByName("glcm")

# Test Image
ii = 0; pp = 0; rr = 0
xind = (ii*7+pp)*(dimX // phantomsInX)
yind = rr*(dimY // phantomsInY)

volume = phanHU[xind:xind+100,yind:yind+100,:]
volumeSITK = sitk.GetImageFromArray(volume)
maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))

featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")

phanFeatures = np.zeros((21,14,len(featureVector)))
featureNames = featureVector.keys()

#%% Skip

for ii, isoval in enumerate(isobvtv_vals): # 3
    
    dilationRadiusArray = isobvtv_xyeval[ii][:,0]
    NseedsArray = isobvtv_xyeval[ii][:,1].astype(int)
    
    for pp in range(samplesPerVal): # 7
        
        for rr, randState in enumerate(randStates): # 14
            
            print(ii, pp, rr)
            
            xind = (ii*7+pp)*(dimX // phantomsInX)
            yind = rr*(dimY // phantomsInY)
            
            volume = phanHU[xind:xind+100,yind:yind+100,:]
            
            volumeSITK = sitk.GetImageFromArray(volume)
            maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
            
            featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
            
            phanFeatures[ii*7+pp,rr] = [featureVector[featureName].item() for featureName in featureVector.keys()]
            
np.save(out_dir+"PhantomX_20210623_features", phanFeatures)

#%%

phanFeatures = np.load(out_dir+"PhantomX_20210623_features.npy")

#%% Skip

import matplotlib.pyplot as plt
import numpy as np

for fnind, fn in enumerate(featureNames):

    plt.figure()
    ax = plt.axes([0, 0.05, 0.9, 0.9])
    im = ax.imshow(phanFeatures[:,:,fnind].T,cmap="plasma")
    plt.axis("off")
    plt.title(fn)
    # cax = plt.axes([0.95, 0.05, 0.05,0.9 ])
    plt.colorbar(mappable=im)
    # plt.savefig(out_dir+'features/'+fn+".png", bbox_inches = 'tight',
    #     pad_inches = 0)
    plt.savefig(out_dir+'features/'+fn+".png", bbox_inches = 'tight')
    
    plt.close("all")
    
#%% Some useful utilities

def getFeaturesContaining(featureString):
    """
    Get Features from phanFeatures and featureNames Containing A Specific Substring
    
    phanFeatures
    featureNames
    isobvtv_xyeval
    
    """
    
    pf = phanFeatures[:,:,np.where([featureString in fn for fn in featureNames])[0]]
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

def plotFeaturesVsRadius(features, nrows, ncols):
    fig = plt.figure()
    for ff, fn in enumerate(features[1]):
        
        f = features[0][:,:,ff].flatten()
        b = features[2][:,:,ff].flatten()
        r = features[3][:,:,ff].flatten()
        n = features[4][:,:,ff].flatten()
        
        mfit, bfit = np.polyfit(r, f, 1)
        r2 = np.corrcoef(r,f)[0,1]**2
    
        plt.subplot(nrows, ncols, ff+1)
        plt.plot(r, f, 'ko', markersize=2)
        plt.plot(r, mfit*r + bfit, "b--")
        plt.title("r2="+"{:.3f}".format(r2),size=8)
        plt.ylabel(fn,size=8)
        
    fig.subplots_adjust(hspace=.35, wspace=.45)
    
    return fig
    
def plotFeaturesVsNseeds(features, nrows, ncols):
    fig = plt.figure()
    for ff, fn in enumerate(features[1]):
        
        f = features[0][:,:,ff].flatten()
        b = features[2][:,:,ff].flatten()
        r = features[3][:,:,ff].flatten()
        n = features[4][:,:,ff].flatten()
        
        mfit, bfit = np.polyfit(n, f, 1)
        r2 = np.corrcoef(n, f)[0,1]**2
    
        plt.subplot(nrows, ncols, ff+1)
        plt.plot(n, f, 'ko', markersize=2)
        plt.plot(n, mfit*n + bfit, "b--")
        plt.title("r2="+"{:.3f}".format(r2),size=8)
        plt.ylabel(fn,size=8)
        
    fig.subplots_adjust(hspace=.35, wspace=.45)
    
    return fig
    
#%%

featuresFO = getFeaturesContaining("original_firstorder")
featuresGLCM = getFeaturesContaining("original_glcm")
featuresGLDM = getFeaturesContaining("original_gldm")
featuresGLRLM = getFeaturesContaining("original_glrlm")

fig = plotFeaturesVsRadius(featuresGLCM, 4, 6)
fig = plotFeaturesVsNseeds(featuresGLCM, 4, 6)

fig = plotFeaturesVsRadius(featuresGLDM, 4, 6)
fig = plotFeaturesVsNseeds(featuresGLDM, 4, 6)

fig = plotFeaturesVsRadius(featuresGLRLM, 4, 6)
fig = plotFeaturesVsNseeds(featuresGLRLM, 4, 6)

#%% Mechanical strength

# Transposed with original phantom
phanHUgrid = np.zeros((100,100,100,len(randStates),samplesPerVal*len(isobvtv_vals)))

for ii, isoval in enumerate(isobvtv_vals): # 3
    
    dilationRadiusArray = isobvtv_xyeval[ii][:,0]
    NseedsArray = isobvtv_xyeval[ii][:,1].astype(int)
    
    for pp in range(samplesPerVal): # 7
        
        for rr, randState in enumerate(randStates): # 14
            
            print(ii, pp, rr)
            
            xind = (ii*7+pp)*(dimX // phantomsInX)
            yind = rr*(dimY // phantomsInY)
            
            # Note: The grid here is transposed
            yind_volumes = ii*7+pp # 0 to 20
            xind_volumes = rr # 0 to 13
            
            volume = phanHU[xind:xind+100,yind:yind+100,:]
            
            phanHUgrid[:,:,:,xind_volumes,yind_volumes] = volume
            
#%% FEA

import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.phantoms.TrabeculaeVoronoi import *
import numpy as np
import matplotlib.pyplot as plt
# import mcubes
from bonebox.FEA.fea import *

rhoBone = 2e-3 # g/mm3
voxelSize = (0.05, 0.05, 0.05) # mm
pixelSize = (0.05, 0.05) # mm
radiusTBS = 5 # pixels
plattenThicknessVoxels = 5 # voxels
plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm    

def computeFEA(volume):

    roiBone = addPlatten(volume, plattenThicknessVoxels)
    vertices, faces, normals, values = Voxel2SurfMesh(roiBone, voxelSize=(0.05,0.05,0.05), step_size=1)
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    nodes, elements, tet = Surf2TetMesh(vertices, faces, verbose=0)
    feaResult = computeFEACompressLinear(nodes, elements, plattenThicknessMM)
    elasticModulus = computeFEAElasticModulus(feaResult)
    
    return elasticModulus

phanStiffness = np.zeros((len(randStates),samplesPerVal*len(isobvtv_vals)))

for xx in range(14):
    for yy in range(21):
        print(xx,yy)
        E = computeFEA(phanHUgrid[:,:,:,xx,yy])
        phanStiffness[xx,yy] = E
        
np.save(out_dir+"PhantomX_20210623_stiffness", phanStiffness)

#%%

phanStiffness = np.load(out_dir+"PhantomX_20210623_stiffness.npy") # so far only the first 4 rows

#%% Correlate stiffness with features

pstiffness = -phanStiffness.T[:,:4].reshape(84)
pfeatures = phanFeatures[:,:4,:].reshape(84,93)
featureNames = list(featureNames)

radiomics_dir = "/data/BoneBox-out/radiomics vs stiffness/"

pr2 = np.zeros(93)

for ii in range(93):
    
    mfit, bfit = np.polyfit(pfeatures[:,ii],pstiffness, 1)
    pr2[ii] = np.corrcoef(pfeatures[:,ii],pstiffness)[0,1]**2
    
    plt.figure(figsize=(4,3))
    plt.plot(pfeatures[:,ii],pstiffness,'k.')
    plt.title(featureNames[ii]+" r2="+"{:.3f}".format(pr2[ii]))
    plt.plot(pfeatures[:,ii], mfit*pfeatures[:,ii] + bfit, "b--")
    plt.xlabel("Feature Value")
    plt.ylabel("Stiffness")
    
    plt.savefig(radiomics_dir+str(ii)+"_"+featureNames[ii]+"_"+"{:.3f}".format(pr2[ii])+".png")
    plt.close("all")

pr2[np.isnan(pr2)]=0
pr2sind = np.argsort(pr2)[::-1]
    
fig, ax = plt.subplots()
ax.barh(np.arange(93),width=pr2[pr2sind[:93]])
ax.set_yticks(np.arange(93))
ax.set_yticklabels([featureNames[ii] for ii in pr2sind[:93]])
ax.invert_yaxis()
plt.tight_layout()

#%% Heatmap

import seaborn as sns
import pandas as pd

sns.set(font_scale=0.4)

d = pd.DataFrame(data=pfeatures,
                 columns=featureNames)
corr = d.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
g = sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1., vmax=1.,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot_kws={"size": 3})

#%% 

features_norm = pfeatures.copy()
features_norm -= np.mean(pfeatures,axis=0) # center on mean
features_norm /= np.std(pfeatures,axis=0) # scale to standard deviation
features_norm[np.isnan(features_norm)] = 0

#%% Random Forest Grid Search using 5-fold cross validation

roi_vm_mean = pstiffness

plt.close('all')

import random
random.seed(1234)

# non-linear without feature selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'max_depth': [2,4,8,16,32,64,128,256], # 16
        'max_leaf_nodes': [2,4,8,16,32,64,128,256], # 8
        'n_estimators': [10,50,100,150,200]} # 50
            ]

rfr = GridSearchCV(
        RandomForestRegressor(), 
        param_grid, cv = 5,
        scoring = 'explained_variance',
        n_jobs=-1
        )

grid_result = rfr.fit(features_norm, roi_vm_mean)
yTrain_fit_rfr = rfr.predict(features_norm)

rfr_params = {'max_depth': rfr.best_estimator_.max_depth,
              'max_leaf_nodes': rfr.best_estimator_.max_leaf_nodes,
              'n_estimators': rfr.best_estimator_.n_estimators}

rfr_params = {'max_depth': 64,
              'max_leaf_nodes': 128,
              'n_estimators': 150}

print(rfr_params)

sns.set(font_scale=1)

mfit, bfit = np.polyfit(roi_vm_mean, yTrain_fit_rfr, 1)
pr2 = np.corrcoef(roi_vm_mean, yTrain_fit_rfr)[0,1]**2

plt.figure()
plt.plot(roi_vm_mean,yTrain_fit_rfr,'ko')
plt.plot(roi_vm_mean, mfit*roi_vm_mean + bfit, "b--")

# Plot feature importance

importances = rfr.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in rfr.best_estimator_], axis = 0)
plt.figure()
plt.title('Feature importances')
plt.barh(range(20), importances[indices[0:20]], yerr = std[indices[0:20]], align = 'center',log=True)
plt.yticks(range(20), list(featureNames[i] for i in indices[0:20] ), rotation=0)
plt.gca().invert_yaxis()
plt.show()

plt.subplots_adjust(left=0.7,bottom=0.1, right=0.8, top=0.9)

# print('Leading features:', feature_names[indices[0:5]])

#%% Half-split

grid_result = rfr.fit(features_norm[:63,:], roi_vm_mean[:63])
yTest_fit_rfr = rfr.predict(features_norm[63:])

sns.set(font_scale=1)

mfit, bfit = np.polyfit(roi_vm_mean[63:], yTest_fit_rfr, 1)
pr2 = np.corrcoef(roi_vm_mean[63:], yTest_fit_rfr)[0,1]**2

plt.figure()
plt.plot(roi_vm_mean[63:],yTest_fit_rfr,'ko')
plt.plot(roi_vm_mean[63:], mfit*roi_vm_mean[63:] + bfit, "b--")

importances = rfr.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in rfr.best_estimator_], axis = 0)
plt.figure()
plt.title('Feature importances')
plt.barh(range(20), importances[indices[0:20]], yerr = std[indices[0:20]], align = 'center',log=True)
plt.yticks(range(20), list(featureNames[i] for i in indices[0:20] ), rotation=0)
plt.gca().invert_yaxis()
plt.show()

plt.subplots_adjust(left=0.7,bottom=0.1, right=0.8, top=0.9)

# ax = sns.heatmap(
#     corr, 
#     vmin=-1, vmax=1, center=0,
#     cmap=sns.diverging_palette(20, 220, n=200),
#     square=True
# )
# ax.set_xticklabels(
#     ax.get_xticklabels(),
#     rotation=45,
#     horizontalalignment='right'
# );

#%% Correlation Feature Value vs Density
# fig = plt.figure()
# for ff, fn in enumerate(featureNamesGLCM):
    
#     f = phanFeaturesGLCM[:,:,ff].flatten()
#     b = bvtvVal[:,:,ff].flatten()
#     r = radiusVal[:,:,ff].flatten()
#     n = NseedsVal[:,:,ff].flatten()
    
#     mfit, bfit = np.polyfit(n, f, 1)
#     r2 = np.corrcoef(n,f)[0,1]**2

#     plt.subplot(4,6,ff+1)
#     plt.plot(n, f, 'ko', markersize=2)
#     plt.plot(n, mfit*n + bfit, "b--")
#     plt.title("r2="+"{:.3f}".format(r2),size=8)
#     # plt.xlabel("R",size=8)
#     plt.ylabel(fn,size=8)
    
# fig.subplots_adjust(hspace=.35, wspace=.45)

#%% Correlation Feature Value vs Radius
# fig = plt.figure()
# for ff, fn in enumerate(featureNamesGLCM):
    
#     f = phanFeaturesGLCM[:,:,ff].flatten()
#     b = bvtvVal[:,:,ff].flatten()
#     r = radiusVal[:,:,ff].flatten()
#     n = NseedsVal[:,:,ff].flatten()
    
#     mfit, bfit = np.polyfit(r, f, 1)
#     r2 = np.corrcoef(r,f)[0,1]**2

#     plt.subplot(4,6,ff+1)
#     plt.plot(r, f, 'ko', markersize=2)
#     plt.plot(r, mfit*r + bfit, "b--")
#     plt.title("r2="+"{:.3f}".format(r2),size=8)
#     # plt.xlabel("R",size=8)
#     plt.ylabel(fn,size=8)
    
# fig.subplots_adjust(hspace=.35, wspace=.45)

# phanHU = np.delete(phanHU,obj=np.arange(3171,3188),axis=0)

# #%% Shuffle - Roll Matrix Positions

# for rr, randState in enumerate(randStates):
    
#     yind = rr*(dimY // phantomsInY)
    
#     phanHU[:,yind:yind+100,:] = np.roll(phanHU[:,yind:yind+100,:], shift=rr*7*(dimX // phantomsInX), axis=0)
    
# np.save(out_dir+"PhantomX_20210623_shuffle.npy", phanHU)

# #%% save shuffled image

# import matplotlib.pyplot as plt
# import numpy as np
# plt.imshow(phanHU[:,:,50].T,cmap="gray")
# plt.axis("off")
# plt.savefig(out_dir+'PhantomX_20210623_shuffle.png', bbox_inches = 'tight',
#     pad_inches = 0)

# #%%

# from PIL import Image
# import numpy as np

# data = phanHU[:,:,50]

# data[data==1500] = 255
# data[data==-50] = 50
# data[data==-800] = 0

# data = data.astype(np.uint8)

# #Rescale to 0-255 and convert to uint8
# # rescaled = (255.0 / data.max() * (data - (data.min()))).astype(np.uint8)

# im = Image.fromarray(rescaled)
# im.save(out_dir+'PhantomX_20210623.png')

# # for xx in range(phantomsInX):
# #     for yy in range(phantomsInY):

#%%

# for rr, randState in enumerate(all_randState):
#     for nn, Nseeds in enumerate(all_Nseeds):
#         for ra, dilationRadius in enumerate(all_dilationRadius):
            
#             print(rr,nn,ra)

#             volume, bvtv, edgeVerticesRetain = makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState)
#             np.save(phantoms_dir+"phan_"+str(rr)+"_"+str(nn)+"_"+str(ra), volume)
#             # E = computeFEA(volume)
            
#             bvtvs[rr,nn,ra] = bvtv
#             # Es[rr,nn,ra] = E
            
# np.save(out_dir+"FEA_bvtvs_2", bvtvs)
# # np.save("FEA_Es_2", Es)

#%%

# bvtvs = np.load(out_dir+"FEA_bvtvs_2.npy")

# import scipy.ndimage

# # plot mean and Std of bvtvs
# meanbvtv = scipy.ndimage.zoom(np.mean(bvtvs,axis=0), 10, order=1)
# extent=[np.min(all_dilationRadius),np.max(all_dilationRadius),
#         np.min(all_Nseeds),np.max(all_Nseeds)]

# fig, ax = plt.subplots()
# ax.imshow(meanbvtv, interpolation="nearest",extent=extent,aspect='auto', origin='lower')
# cs = ax.contour(meanbvtv, extent=extent, colors='w', origin='lower')
# ax.clabel(cs, fontsize=9, inline=True)

# plt.xlabel("Dilation Radius")
# plt.ylabel("N")

# # Get contour lines
# samplesPerVal = 4
# isobvtv_vals = [0.32]
# isobvtv_xy = []
# isobvtv_xyeval = []

# for vv, isobvtv in enumerate(isobvtv_vals):
#     cs = ax.contour(meanbvtv, [isobvtv], extent=extent)
#     p = cs.collections[0].get_paths()[0]
#     isobvtv_xy.append(p.vertices)
    
# for ii, xy in enumerate(isobvtv_xy):
    
#     xylist = []
    
#     Nxy = xy.shape[0]
#     Dxy = Nxy // (samplesPerVal-1)
    
#     # select 4 evenly spaced points along the isocline
#     for ss in range(samplesPerVal-1):
#         xylist.append(xy[ss*Dxy,:])
#     xylist.append(xy[-1,:])
    
#     isobvtv_xyeval.append(np.array(xylist))
    
#%%

# #%% Export Triplanar images of the phantoms

# for rr, randState in enumerate(all_randState):
#     for nn, Nseeds in enumerate(all_Nseeds):
#         for ra, dilationRadius in enumerate(all_dilationRadius):
            
#             volume = np.load(phantoms_dir+"phan_"+str(rr)+"_"+str(nn)+"_"+str(ra)+".npy")
#             plt.subplot(1, 3, 1)
#             plt.imshow(volume[:,:,50],interpolation="nearest",cmap="gray")
#             plt.title("XY"); plt.axis("off")
#             plt.subplot(1, 3, 2)
#             plt.imshow(volume[:,50,:],interpolation="nearest",cmap="gray")
#             plt.title("XZ"); plt.axis("off")
#             plt.subplot(1, 3, 3)
#             plt.imshow(volume[50,:,:],interpolation="nearest",cmap="gray")
#             plt.title("YZ"); plt.axis("off")
            
#             plt.savefig(phantoms_dir+"fig_"+str(rr)+"_"+str(nn)+"_"+str(ra))
#             plt.close("all")

# #%% Evaluates points along isobvtv_xyeval

# phantoms_iso_dir = "/data/BoneBox-out/phantoms_iso/"

# randStates = [2, 3]
# EsArr= np.zeros((len(isobvtv_vals),samplesPerVal,len(randStates)))

# for ii, isoval in enumerate(isobvtv_vals):
    
#     dilationRadiusArray = isobvtv_xyeval[ii][:,0]
#     NseedsArray = isobvtv_xyeval[ii][:,1].astype(int)
    
#     for pp in range(samplesPerVal):
#         for rr, randState in enumerate(randStates):
            
#             print(ii, pp, rr)
            
#             volume, bvtv, edgeVerticesRetain = makePhantom(dilationRadiusArray[pp],
#                                                            NseedsArray[pp],
#                                                            edgesRetainFraction,
#                                                            randState)
            
#             np.save(phantoms_iso_dir+"phan_"+str(ii)+"_"+str(pp)+"_"+str(rr), volume)
#             plt.subplot(1, 3, 1)
#             plt.imshow(volume[:,:,50],interpolation="nearest",cmap="gray")
#             plt.title("XY"); plt.axis("off")
#             plt.subplot(1, 3, 2)
#             plt.imshow(volume[:,50,:],interpolation="nearest",cmap="gray")
#             plt.title("XZ"); plt.axis("off")
#             plt.subplot(1, 3, 3)
#             plt.imshow(volume[50,:,:],interpolation="nearest",cmap="gray")
#             plt.title("YZ"); plt.axis("off")
#             plt.savefig(phantoms_iso_dir+"fig_"+str(ii)+"_"+str(pp)+"_"+str(rr))
#             plt.close("all")
            
#             E = computeFEA(volume)
#             EsArr[ii,pp,rr] = E
            
#             print(E)
            
# np.save(out_dir+"EsArr", EsArr)

# #%%



# #%%

# vol = np.ones(volume.shape).astype(bool)
# vol[0,:,:] = False; vol[-1,:,:] = False; 
# vol[:,0,:] = False; vol[:,-1,:] = False; 
# vol[:,:,0] = False; vol[:,:,-1] = False; 

# E0 = computeFEA(vol)

# #%%

# a_strings = ["%.2f" % x for x in dilationRadiusArray]
# plt.boxplot(np.squeeze(EsArr).T/E0)
# plt.xticks([1, 2, 3, 4], a_strings)
# plt.grid()

#%%

# EsArray = np.array(EsList)

# bvtvs = np.load(out_dir+"FEA_bvtvs_2.npy")

# Es = np.load("FEA_Es_2.npy")

# plt.plot(bvtvs.flatten(), -Es.flatten(),'ko')

# # np.save("FEAbvtvs",bvtvs)
# # np.save("FEAEs",Es)

# plt.imshow(np.mean(bvtvs,axis=0))
# plt.axis("off")
# plt.colorbar()

# plt.imshow(np.std(bvtvs,axis=0))
# plt.axis("off")
# plt.colorbar()

# plt.imshow(np.mean(Es,axis=0))
# plt.axis("off")
# plt.colorbar()

# plt.imshow(np.std(Es,axis=0))
# plt.axis("off")
# plt.colorbar()

# plt.plot(bvtvs.flatten(),Es.flatten(),'ko')