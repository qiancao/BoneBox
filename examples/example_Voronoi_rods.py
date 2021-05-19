# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:55:11 2021

@author: Qian.Cao

Generate a series of Voronoi Rod phantoms

"""

import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.phantoms.TrabeculaeVoronoi import *
import numpy as np
import matplotlib.pyplot as plt
# import mcubes

plt.ion()

print('Running example for TrabeculaeVoronoi')

def makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState):

    # Parameters for generating phantom mesh
    Sxyz, Nxyz = (10,10,10), (Nseeds, Nseeds, Nseeds) # volume extent in XYZ (mm), number of seeds along XYZ
    Rxyz = 1.
    # edgesRetainFraction = 0.5
    facesRetainFraction = 0.5
    # dilationRadius = 3 # (voxels)
    # randState = 123 # for repeatability
    
    # Parameters for generating phantom volume
    volumeSizeVoxels = (200,200,200)
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
    
    return volumeDilated, bvtv, edgeVerticesRetain
    
    # Visualize all edges
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for ii in range(edgeVertices.shape[0]):
    #     ax.plot(edgeVertices[ii,:,0],edgeVertices[ii,:,1],edgeVertices[ii,:,2],'b-')
    
    # volumeSmoothed = mcubes.smooth(volumeDilated)
    
rhoBone = 2e-3 # g/mm3
voxelSize = (0.05, 0.05, 0.05) # mm
pixelSize = (0.05, 0.05) # mm
radiusTBS = 5 # pixels

def computeROIProjection(roiBone, projectionAxis):
    projectionImage = np.prod(np.array(voxelSize)) * rhoBone * np.sum(roiBone,axis=projectionAxis).T \
         / np.prod(np.array(pixelSize))
    return projectionImage

#%%

# Projection/TBS Settings
dilationRadius = 3.4
randState = 1
Nseeds = 12
edgesRetainFraction = 0.5

volume, bvtv, edgeVerticesRetain = makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState)
projection = computeROIProjection(volume, 0)

plt.figure()
plt.imshow(projection, cmap="gray")
plt.axis("off")
plt.title("BMD/BvTv: "+"{0:.3g}".format(bvtv))
plt.clim(0,0.012)
plt.colorbar()

#%% Visualize all edges

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for ii in range(edgeVerticesRetain.shape[0]):
    ax.plot(edgeVerticesRetain[ii,:,0],edgeVerticesRetain[ii,:,1],
            edgeVerticesRetain[ii,:,2],'b-')
    
#%% Look at impact to bvtv

radii = np.linspace(1,3,10)
ns = range(5,18)

edgesRetainFraction = 0.5

bvtvs = np.zeros((len(radii),len(ns)))

for rr in range(len(radii)):
    for nn in range(len(ns)):
        print(str(rr) + " " + str(nn))
        volume, bvtv, edgeVerticesRetain = makePhantom(radii[rr], ns[nn], edgesRetainFraction, randState)
        bvtvs[rr,nn] = bvtv