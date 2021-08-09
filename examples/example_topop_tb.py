#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 02:58:30 2021

@author: qcao
"""

import sys
sys.path.append('../') # use bonebox from source without having to install/build

import numpy as np
import matplotlib.pyplot as plt

from bonebox.phantoms.TrabeculaeVoronoi import *
from bonebox.FEA.fea import *

from skimage.morphology import closing

if __name__ == "__main__":
    
    # Parameters for generating phantom mesh
    Sxyz, Nxyz = (10,10,10), (10,10,10) # volume extent in XYZ (mm), number of seeds along XYZ
    Rxyz = 1.
    edgesRetainFraction = 0.5
    facesRetainFraction = 0.1
    dilationRadius = 3 # (voxels)
    randState = 123 # for repeatability
    
    morphClosingMask = np.ones((3,3,3)) # mask for morphological closing
    
    # Parameters for generating phantom volume
    volumeSizeVoxels = (200,200,200)
    voxelSize = np.array(Sxyz) / np.array(volumeSizeVoxels)
    
    # FEA parameters
    plattenThicknessVoxels = 5 # voxels
    
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
    
    # Make edge volume
    uniqueEdgesRetain = uniqueEdges
    volumeEdges = makeSkeletonVolumeEdges(vor.vertices, uniqueEdgesRetain, voxelSize, volumeSizeVoxels)
    # volumeFaces = makeSkeletonVolumeFaces(vor.vertices, uniqueFacesRetain, voxelSize, volumeSizeVoxels)
    
    # Morphological closing on volume of edges
    # volumeSkeleton = closing(np.logical_or(volumeEdges,volumeFaces),morphClosingMask)
    volumeSkeleton = closing(volumeEdges, morphClosingMask)
    volume = dilateVolumeSphereUniform(volumeSkeleton, dilationRadius)
    
    # Add platten
    volume = addPlatten(volume, plattenThicknessVoxels)
    
    # Convert to FEA mesh
    nodes, elements = Voxel2HexaMeshIndexCoord(volume)
    nodes = Index2AbsCoords(nodes, volumeSizeVoxels, voxelSize=voxelSize)
    
    