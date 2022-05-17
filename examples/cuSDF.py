#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:27:01 2022

@author: qcao

CuPy-based implementation of signed distance fields

TODO: NOT memory-efficient, try kernel implementation in the future

Resources:
https://github.com/fogleman/sdf
https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
https://iquilezles.org/www/articles/smin/smin.htm
Capsult / Line - exact
Triangle - exact

"""

import numpy as np
import cupy as cp
from cupyx.profiler import benchmark
import matplotlib.pyplot as plt

import sys
sys.path.append("../bonebox/phantoms/") # from examples folder
sys.path.append("../bonebox/FEA/") # from examples folder

from fea import *
from vtk_utils import *
from TrabeculaeVoronoi import *

# meshing
from skimage import measure
import tetgen
import trimesh
import pyvista as pv

def sdf_coordinates(dims, spacing):
    
    # Array coordinates
    xxx, yyy, zzz = np.meshgrid(np.arange(dims[0]),np.arange(dims[1]),np.arange(dims[2]), indexing="ij")
    xyz = np.vstack((xxx.flatten(),yyy.flatten(),zzz.flatten())).T
    
    # Convert to world coordinates
    xyz = convertArray2Abs(xyz, spacing, dims)

    # Transfer to GPU
    xyz = cp.array(xyz.astype(np.float32))
    
    return xyz

def dot2(v):
    return cp.sum(v*v,axis=-1)

def mix(x,y,a):
    return x*(1-a) + y*a

def sdf_line(p,a,b,r):
    
    pa = p-a
    ba = b-a
    dot_baba = cp.dot(ba,ba)
    h = cp.clip( cp.dot(pa,ba) / dot_baba, 0., 1. )
    sdf = cp.linalg.norm(pa - ba*h[:,None],axis=1) - r
    
    return sdf

def sdf_triangle(p, a, b, c, r):
    
    ba = b-a
    pa = p-a
    cb = c-b
    pb = p-b
    ac = a-c
    pc = p-c
    nor = cp.cross(ba,ac)
    
    # true for point-edge distance, false for point-face distance
    in_out = ((cp.sign(cp.inner(cp.cross(ba,nor),pa)) \
               + cp.sign(cp.inner(cp.cross(cb,nor),pb)) \
               + cp.sign(cp.inner(cp.cross(ac,nor),pc))) < 2.).astype(int)
    
    sdf = cp.sqrt(in_out*cp.minimum(cp.minimum(dot2(ba[None,:]*cp.clip(cp.inner(ba,pa)/dot2(ba),0.,1.)[:,None]-pa),
                                 dot2(cb[None,:]*cp.clip(cp.inner(cb,pb)/dot2(cb),0.,1.)[:,None]-pb)),
                            dot2(ac[None,:]*cp.clip(cp.inner(ac,pc)/dot2(ac),0.,1.)[:,None]-pc)) \
                  + cp.logical_not(in_out) * (cp.inner(nor,pa) * cp.inner(nor,pa) / dot2(nor))) - r
        
    return sdf

def sdf_smooth_union(d1,d2,k=5): # TODO: intiialization-dependent? check this
    
    h = cp.clip(0.5 + 0.5*(d2-d1)/k, 0., 1.)
    sdf = mix(d2,d1,h) - k*h*(1.-h)
    
    return sdf

def sdf_smooth_union_exp(d1,d2,k=5):
    
    h = cp.exp2(-k*d1) + cp.exp2(-k*d2)
    sdf = -log2(h)/k
    
    return sdf

def sdf_union(d1,d2):
    
    sdf = cp.minimum(d1,d2)
    
    return sdf

def sdf_cupy_to_numpy(sdf, dims):
    
    sdf = cp.asnumpy(sdf.reshape(dims))
    
    # difference between numpy array coordinates and world coordinates?
    
    return sdf

def set_sdf_boundary(sdf,value=None):
    # expects a unflattened array
    
    if value is None:
        value = np.max(sdf)
    
    sdf[0,:,:] = value
    sdf[-1,:,:] = value
    sdf[:,0,:] = value
    sdf[:,-1,:] = value
    sdf[:,:,0] = value
    sdf[:,:,-1] = value
    
    return sdf

def sdf_rod_plate(vertices, faces, faces_r, edges, edges_r, dims, spacing, k=2):
    
    xyz = sdf_coordinates(dims, spacing)
    
    vertices = cp.array(vertices,dtype=cp.float32)
    faces = cp.array(faces,dtype=cp.uint32)
    faces_r = cp.array(faces_r,dtype=cp.float32)
    edges = cp.array(edges,dtype=cp.uint32)
    edges_r = cp.array(edges_r,dtype=cp.float32)
    
    # compute the very first line to allocate volume
    # sdf = sdf_triangle(xyz,vertices[faces[0]][0],vertices[faces[0]][1],vertices[faces[0]][2],faces_r[0])
    sdf = cp.empty(int(np.prod(dims)),dtype=cp.float32)
    sdf[:] = cp.Inf
    
    for ind, edge in enumerate(edges):
        sdf_tmp = sdf_line(xyz,vertices[edge][0],vertices[edge][1],edges_r[ind])
        sdf = sdf_union(sdf,sdf_tmp)
        # sdf = sdf_smooth_union_exp(sdf,sdf_tmp,k)
        
    for ind, face in enumerate(faces):
        print(f"{ind}/{len(faces)}")
        sdf_tmp = sdf_triangle(xyz,vertices[face][0],vertices[face][1],vertices[face][2],faces_r[ind])
        sdf = sdf_union(sdf,sdf_tmp)
        # sdf = sdf_smooth_union_exp(sdf,sdf_tmp,k)
        
    return sdf_cupy_to_numpy(sdf, dims)

if __name__ == "__main__":
    
    def sdf_show(sdf):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(cp.asnumpy(sdf.reshape(dims)[250,:,:]))
        ax[1].imshow(cp.asnumpy(sdf.reshape(dims)[:,250,:]))
        ax[2].imshow(cp.asnumpy(sdf.reshape(dims)[:,:,250]))
        return fig, ax
    
    # volume (assume volume center as origin)
    dims = (500,500,500)
    spacing = (0.5,0.5,0.5)
    
    # test phantom
    vertices = np.array([[0,50,-50],[0,50,50],[0,0,20],[0,-120,-120],[0,120,120]]).astype(float) # must be integer coordinates!!!
    faces = np.array([[0,1,2],]) # triangles
    faces_r = np.array([5]) # half-thickness for plates
    edges = np.array([[1,4],]) # line segments
    edges_r = np.array([4]) # radii for rods

    sdf = sdf_rod_plate(vertices, faces, faces_r, edges, edges_r, dims, spacing, k=2)
    
    sdf_show(sdf)
    
    sdf = set_sdf_boundary(sdf)
    vertices_sdf, faces_sdf, normals, values = Voxel2SurfMesh(sdf, voxelSize=spacing, origin=None, level=0, step_size=1, allow_degenerate=False)
    tmesh = trimesh.Trimesh(vertices_sdf, faces=faces_sdf)
    surf = pv.wrap(tmesh)
    surf.plot()