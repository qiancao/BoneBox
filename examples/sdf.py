#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Voxelization using signed distance fields (SDF)

assume coordinates are absolute to accomodate anisotropic volumes

SDF volume is np.nan by default, 

Based on test_surface_line_sdf.

Qian Cao

"""

import sys
sys.path.append("../bonebox/phantoms/") # TODO: remove this when incorporated into library

import numpy as np
# from pysdf import SDF
from TrabeculaeVoronoi import convertAbs2Array, convertArray2Abs
from numba import jit

import matplotlib.pyplot as plt
import pyvista as pv

#%% Voxelization

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

def SDF_triangle(vertices, face, dims, spacing, max_sdf, radius=0., modulation=None):
    """
    
    Computes the absolute value of SDF (unsigned distance function) for a triangular mesh.
    
    Inputs:
        vertices: coordinates in world coordinates
        face: triangular mesh - (3,) integer indices
        dims: number of voxels in X, Y, and Z directions
        spacing: voxel size in world coordinate units
        max_sdf: threshold for maximum SDF computed (skips other voxels)
        scaling: sdf is multiplied by this value, multiply by inverse of radius to change thickness
        modulation: if provided a function from 0 to 1
    
    Returns volume of SDF:
        np.NaN - not yet visited
        np.Inf - visited, SDF value > thresh_sdf
    
    """
    
    # assert face.shape==(3,), "face.shape must be (3,)"
    #TODO: What if 3 points coplanar, is this possible?
    
    # SDF volume initialized with NaN (not visited)
    volume = np.empty(dims,dtype=float)
    volume[:] = np.nan
    
    # Initial query points: centroid and vertices
    centroid = np.mean(vertices[face,:],axis=0)
    points = np.vstack((centroid,vertices[face,:]))
    
    # convert absolute coordinates to array coordinates
    xyz = np.round(convertAbs2Array(points, spacing, dims)).astype(int)
    
    # compute SDF values
    f = SDF(vertices, face, robust=True)
    
    # for it in range(15):
    while len(xyz) > 0:
        
        # convert array coordinates to absolute coordinates
        points = convertArray2Abs(xyz, spacing, dims)
        
        # compute sdf values for points
        # sdf = np.abs( np.maximum(f.calc(points) - radius, 0) ) 
        sdf = np.abs(f.calc(points)) - radius
        
        # voxels with sdf values above threshold is no longer used to find neighbors for the next iteration
        sdf[sdf>max_sdf] = np.inf
        
        # assign sdf values to volume
        volume[tuple(xyz.T)] = sdf
        
        # get the bext batch of coordinates
        xyz = get_valid_neighbors(xyz,volume)
        
    # Set high SDF voxels to np.nan
    volume[np.isinf(volume)] = np.nan
        
    return volume

def SDF_triangle_inplace(volume, vertices, face, dims, spacing, max_sdf, radius=0., modulation=None):
    """
    
    Computes the absolute value of SDF (unsigned distance function) for a triangular mesh.
    
    Inputs:
        vertices: coordinates in world coordinates
        face: triangular mesh - (3,) integer indices
        dims: number of voxels in X, Y, and Z directions
        spacing: voxel size in world coordinate units
        max_sdf: threshold for maximum SDF computed (skips other voxels)
        scaling: sdf is multiplied by this value, multiply by inverse of radius to change thickness
        modulation: if provided a function from 0 to 1
    
    Returns volume of SDF:
        np.NaN - not yet visited
        np.Inf - visited, SDF value > thresh_sdf
        
    Same as SDF_triangle but merge
    
    """
    
    # assert face.shape==(3,), "face.shape must be (3,)"
    #TODO: What if 3 points coplanar, is this possible?
    
    # SDF volume initialized with NaN (not visited)
    # volume = np.empty(dims,dtype=float)
    # volume[:] = np.nan
    
    # Initial query points: centroid and vertices
    centroid = np.mean(vertices[face,:],axis=0)
    points = np.vstack((centroid,vertices[face,:]))
    
    # convert absolute coordinates to array coordinates
    xyz = np.round(convertAbs2Array(points, spacing, dims)).astype(int)
    
    # compute SDF values
    f = SDF(vertices, face, robust=True)
    
    # for it in range(15):
    while len(xyz) > 0:
        
        # convert array coordinates to absolute coordinates
        points = convertArray2Abs(xyz, spacing, dims)
        
        # compute sdf values for points
        # sdf = np.abs( np.maximum(f.calc(points) - radius, 0) ) 
        sdf = np.abs(f.calc(points)) - radius
        
        # voxels with sdf values above threshold is no longer used to find neighbors for the next iteration
        sdf[sdf>max_sdf] = np.inf
        
        # assign sdf values to volume
        volume[tuple(xyz.T)] = sdf
        
        # get the bext batch of coordinates
        xyz = get_valid_neighbors(xyz,volume)
        
    # Set high SDF voxels to np.nan
    volume[np.isinf(volume)] = np.nan
        
    return volume

def SDF_plate(vertices, face, dims, spacing, max_sdf, radius=0, modulation=None):
    """
    TODO: NOT YET IMPLEMENTED
    
    Computes the absolute value of SDF (unsigned distance function) for a plate broken down into triangular meshes.
    
    Inputs:
        vertices: coordinates in world coordinates
        faces: triangular mesh - (3,) integer indices
        dims: number of voxels in X, Y, and Z directions
        spacing: voxel size in world coordinate units
        max_sdf: threshold for maximum SDF computed (skips other voxels)
        scaling: sdf is multiplied by this value, multiply by inverse of radius to change thickness
        modulation: if provided a function from 0 to 1
    
    
    Returns volume of SDF:
        np.NaN - not yet visited
        np.Inf - visited, SDF value > thresh_sdf
    
    """
    
    # assert face.ndim==1, "Dimension of face must be 1, it is an array of vertex indices, in correct rotation order"
    # assert np.max(face) < vertices.shape[0], "Largest index in face exceeds number of vertices"
    
    assert False # TODO this function is not yet implemented
    
    # SDF volume initialized with NaN (not visited)
    volume = np.empty(dims,dtype=float)
    volume[:] = np.nan
    
    # Initial query points: centroid and vertices
    centroid = np.mean(vertices[face,:],axis=0)
    points = np.vstack((centroid,vertices[face,:]))
    
    # convert absolute coordinates to array coordinates
    xyz = np.round(convertAbs2Array(points, spacing, dims)).astype(int)
    
    # compute SDF values
    f = SDF(vertices, face, robust=False)
    
    # for it in range(15):
    while len(xyz) > 0:
        
        # convert array coordinates to absolute coordinates
        points = convertArray2Abs(xyz, spacing, dims)
        
        # compute sdf values for points
        sdf = np.abs(np.maximum(f.calc(points) - radius, 0))
        
        # voxels with sdf values above threshold is no longer used to find neighbors for the next iteration
        sdf[sdf>max_sdf] = np.inf
        
        # assign sdf values to volume
        volume[tuple(xyz.T)] = sdf
        
        # get the bext batch of coordinates
        xyz = get_valid_neighbors(xyz,volume)
        
    # Set high SDF voxels to np.nan
    volume[np.isinf(volume)] = np.nan
        
    return volume

def SDF_linesegment(vertices, line, dims, spacing, max_sdf, radius=0., modulation=None):
    """
    
    Computes the absolute value of SDF (unsigned distance function) for a line segment.
    
    Inputs:
        vertices: coordinates in world coordinates
        line: (2,) integer indices 
        dims: number of voxels in X, Y, and Z directions
        spacing: voxel size in world coordinate units
        max_sdf: threshold for maximum SDF computed (skips other voxels)
        scaling: sdf is multiplied by this value, multiply by inverse of radius to change thickness
        modulation: if provided a function from 0 to 1
    
    
    Returns volume of SDF:
        np.NaN - not yet visited
        np.Inf - visited, SDF value > thresh_sdf (this is returned)
        
    # Reference: https://github.com/sxyu/sdf/blob/master/include/sdf/sdf.hpp
    
    """
    
    # assert line.shape==(2,), "line.shape must be (2,)"
    
    A = vertices[line[0],:]
    B = vertices[line[1],:]
    AB = B - A
    AB_sq_norm = AB/np.sum(AB**2) # AB / norm(AB)**2
    
    # SDF volume initialized with NaN (not visited)
    volume = np.empty(dims,dtype=float)
    volume[:] = np.nan
    
    # Initial query points: centroid and vertices
    centroid = np.mean(vertices[line,:],axis=0)
    points = np.vstack((centroid,vertices[line,:]))
    
    # convert absolute coordinates to array coordinates
    xyz = np.round(convertAbs2Array(points, spacing, dims)).astype(int)
    
    # for it in range(15):
    while len(xyz) > 0:
        
        # convert array coordinates to absolute coordinates
        points = convertArray2Abs(xyz, spacing, dims)
        
        # compute sdf values for points
        AP = points - A
        t = np.dot(AP,AB_sq_norm)[:,None]
        t = np.maximum(0.,np.minimum(t,1.))
        # sdf = np.maximum(np.sqrt(np.sum((AP - t*AB)**2,axis=1)) - radius, 0)
        sdf = np.sqrt(np.sum((AP - t*AB)**2,axis=1)) - radius
        
        # voxels with sdf values above threshold is no longer used to find neighbors for the next iteration
        sdf[sdf>max_sdf] = np.inf
        
        # assign sdf values to volume
        volume[tuple(xyz.T)] = sdf
                
        # get the bext batch of coordinates
        xyz = get_valid_neighbors(xyz,volume)
        
    # Set high SDF voxels to np.nan
    volume[np.isinf(volume)] = np.nan
    
    return volume

def merge_sdf_volumes_min(v0, v1):
    """
    Combine two np.array volumes v0 and v1:
    
    if v0 and v1 voxels are both np.nan, output voxel is np.nan
    if one is not np.nan, output voxel takes the non-nan value
    if both are not nan, output voxel is either min
    
    """
    
    # assert v0.shape == v1.shape, "v1 and v0 to be merged must be of the same size."
    
    v0[np.isinf(v0)] = np.nan
    v1[np.isinf(v1)] = np.nan
    
    volume = np.empty(v0.shape,dtype=float)
    volume[:] = np.nan
    
    # take v0 value when v1 is nan
    ind = np.logical_not(np.isnan(v0)) & np.isnan(v1)
    volume[ind] = v0[ind] 
    
    # take v1 value when v0 is nan
    ind = np.isnan(v0) & np.logical_not(np.isnan(v1))
    volume[ind] = v1[ind]
    
    # take v1 value when v0 is nan
    ind = np.logical_not(np.isnan(v0) | np.isnan(v1))
    volume[ind] = np.minimum(v0[ind], v1[ind])
    
    return volume

def volume_sdf(vertices, faces, face_thicknesses, edges, edge_radii, dims, spacing, max_sdf):
    """
    
    Create volume of SDF
    
    faces -> plates (triangular faces only, see subdivideFacesWithLocalNeighbors(vertices, faces))
    edges -> line segments
    
    SDF normalized such that contour of sdf=-1 corresponds to face_thickneses and edge_radii
    
    """
    
    # assert len(faces) == len(face_thicknesses), "len(faces) must be equal to len(face_thicknesses)"
    # assert len(edges) == len(edge_radii), "len(edges) must be equal to len(edge_radii)"
    
    # Convert thicknesses and radii to scalings
    face_radii = face_thicknesses/2
    edge_radii = edge_radii
    
    volume = np.empty(dims,dtype=float)
    volume[:] = np.nan
    
    for ind, face in enumerate(faces):
        print(f"{ind}/{len(faces)}")
        sdf = SDF_triangle(vertices, face, dims, spacing, max_sdf, radius=face_radii[ind], modulation=None)
        volume = merge_sdf_volumes_min(volume, sdf)
        
    for ind, edge in enumerate(edges):
        sdf = SDF_linesegment(vertices, edge, dims, spacing, max_sdf, radius=edge_radii[ind], modulation=None)
        volume = merge_sdf_volumes_min(volume, sdf)
        
    return volume

if __name__ == "__main__":


    #%% Test surface SDF
    
    plt.close("all")
    
    max_sdf = 5. # maximum sdf labelled (in absolute units)
    
    dims = (500,500,500) # volumeSizeVoxels
    spacing = (1,1,1) # voxelSize
    # origin = -dims/2 # assume volume is centered at ZERO
    
    # triangular mesh ()
    vertices = np.array([[0,50,-50],[0,50,50],[0,0,20]]).astype(float)
    face = np.array([0,1,2])
    v0 = SDF_triangle(vertices, face, dims, spacing, max_sdf, radius = 2)

    
    vertices = np.array([[0,-120,-120],[0,120,120],[120,120,0]]).astype(float)
    line = np.array([0,1])
    v1 = SDF_linesegment(vertices, line, dims, spacing, max_sdf, radius = 3)

    volume = merge_sdf_volumes_min(v0, v1)
    
    fig = plt.figure()
    plt.imshow(volume[250,:,:])