#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 18:25:37 2022

@author: qcao

looking at voxelization algorithms in pysdf and BFS
assume coordinates are absolute to accomodate anisotropic volumes

SDF volume is np.nan by default, 


"""

import sys
sys.path.append("../bonebox/phantoms/")

import numpy as np

import numba
from pysdf import SDF

import matplotlib.pyplot as plt
import pyvista as pv

from TrabeculaeVoronoi import *

#%%

thresh_sdf = 10. # maximum sdf labelled (in absolute units)

dims = (100,100,100) # volumeSizeVoxels
spacing = (1,1,1) # voxelSize
# origin = -dims/2 # assume volume is centered at ZERO

# triangular mesh ()
vertices = np.array([[-40,-40,-40],[20,20,20],[0,0,-30]]).astype(float)
faces = np.array([[0,1,2],])

#%%

def get_valid_neighbors(xyz, volume):
    # finds 6-connected-neighbors of xyz to compute SDF
    # xyz is an Nx3 integer array of voxel coordinates
    
    # get volume dimension
    dims = volume.shape
    
    # filter out np.inf coordinates above thresh_sdf
    values = volume[tuple(xyz.T)]
    xyz = xyz[np.logical_not(np.isinf(values))]
    
    # get neighboring coordinates
    xyz = np.vstack((xyz-(1,0,0), xyz+(1,0,0), 
                     xyz-(0,1,0), xyz+(0,1,0), 
                     xyz-(0,0,1), xyz+(0,0,1)))
    
    # filter out duplicate points
    xyz = np.unique(xyz,axis=0)
    
    # filter out out-of-bounds coordinates
    valid_ind = np.ones(len(xyz),dtype=bool)
    for axis in range(3):
        valid_ind = valid_ind & (xyz.T[axis] > 0) & (xyz.T[axis] < dims[axis])
    xyz = xyz[valid_ind]
    
    # only keep unvisited voxels in neighbors
    values = volume[tuple(xyz.T)]
    xyz = xyz[np.isnan(values)]
    
    return xyz

"""

Volume Encoding:
np.NaN - not yet visited
np.Inf - visited, SDF value > thresh_sdf

"""

# SDF volume initialized with NaN (not visited)
volume = np.empty(dims,dtype=float)
volume[:] = np.nan

# Initial query points: centroid and vertices
centroid = np.mean(vertices,axis=0)
points = np.vstack((centroid,vertices))

# convert absolute coordinates to array coordinates
xyz = np.round(convertAbs2Array(points, spacing, dims)).astype(int)

# compute SDF values
f = SDF(vertices, faces, robust=False)

# for it in range(15):
while len(xyz) > 0:
        
    print(len(xyz))
    
    # convert array coordinates to absolute coordinates
    points = convertArray2Abs(xyz, spacing, dims)
    
    # compute sdf values for points
    sdf = np.abs(f.calc(points))
    
    # voxels with sdf values above threshold is no longer used to find neighbors for the next iteration
    sdf[sdf>thresh_sdf] = np.inf
    
    # assign sdf values to volume
    volume[tuple(xyz.T)] = sdf
    
    # get the bext batch of coordinates
    xyz = get_valid_neighbors(xyz,volume)

# print(np.sum(np.logical_not(np.isnan(volume))))
plt.imshow(volume[:,30,:])