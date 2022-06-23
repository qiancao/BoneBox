#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:37:21 2022

@author: qian.cao

# Run FEA on Harsha's 700+ new ROIs for the L4 L5 data
# does not perform FEA, meshing only

# commandline tool for meshing a numpy array (isosurface, smoothing, simplification, meshfix)
# parallel script to be run with .._partial_driver.py

Inputs:

path to input binary nrrd file
voxel size
platten thickness (in voxels)
path to output stl file
path to output vtk file
verbose

https://stackoverflow.com/questions/11818640/parallel-running-of-several-jobs-in-a-python-script

"""

import numpy as np
import nrrd

import os
import sys

sys.path.append("/home/qian.cao/projectchrono/chrono_build/build/bin") # TODO: pychrono temporary build
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../bonebox/FEA/"))) # TODO: replace this with relative imports, e.g. from ..FEA.fea import *
from fea import addPlatten, set_volume_bounds, filter_connected_volume, \
    Voxel2SurfMesh, isWatertight, smoothSurfMesh, \
        simplifySurfMeshACVD, repairSurfMesh, Surf2TetMesh, saveTetrahedralMesh\

import argparse

def cropCubeFromCenter(img,length):
    
    x0,y0,z0 = np.array(img.shape)//2
    R = length//2
    
    return img[slice(x0-R,x0+R+1),
               slice(y0-R,y0+R+1),
               slice(z0-R,z0+R+1)]

if __name__ == "__main__":

    # Parse arguments
    parser = argparse.ArgumentParser(description="Process binary images (.nrrd) into tetrahedral meshes (.vtk)")
    parser.add_argument("filepath_nrrd",type=str,help="string, path to input nrrd file")
    parser.add_argument("filepath_vtk",type=str,help="string, path to ouput vtk file")
    parser.add_argument("--filepath_log",type=str,default=None, help="string, path to log file")
    parser.add_argument("--crop_size",type=int,default=None,help="if set, crops volume at the center before meshing (voxels)")
    parser.add_argument("--voxel_size",type=float,default=None,help="float, if set, overwrites nrrd voxel size, assumes isotropic")
    parser.add_argument("--platten_thickness",type=int,default=10,help="thickness of compression plattens, 10 voxels if not set")
    parser.add_argument("--smooth_iterations",type=int,default=15,help="integer, iterations of smoothing to perform on surface mesh")
    parser.add_argument("--simplify_fraction",type=float,default=0.15,help="float, fraction of mesh vertices to keep in ACVD")
    args = parser.parse_args()

    # Input Output Names
    filenameNRRD = args.filepath_nrrd
    filenameVTK = args.filepath_vtk

    # Read NRRD file
    try:
        volume, header = nrrd.read(filenameNRRD)
    except:
        print("nrrd.read unable to read file")
    
    # Set voxel size
    if (args.voxel_size is not None) or ("spacings" not in header.keys()): # use prespecified voxel size
        voxelSize = (args.voxel_size,)*3 # mm
    else:
        voxelSize = header["spacings"]

    # Set image size (to crop from the input image)
    if args.crop_size is not None:
        volume = cropCubeFromCenter(volume,args.crop_size)
        cubeShape = volume.shape
    else:
        cubeShape = args.crop_size

    # Set FEA platten thickness
    plattenThicknessVoxels = args.platten_thickness # voxels
    plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm

    STEPSIZE = 1

    # Make output directories
    vtkDir, vtkTail = os.path.split(args.filepath_vtk)
    os.makedirs(vtkDir,exist_ok=True)

    # logDir, logTail = os.path.split(args.filepath_log)
    # os.makedirs(logDir,exist_ok=True)
    
    # Read NRRD file
    try:

        volume = cropCubeFromCenter(volume,cubeShape[0]) # crop the ROI to a 1 cm3 volume
        volume = addPlatten(volume, plattenThicknessVoxels)
        volume = set_volume_bounds(volume, airValue=None,bounds=2) # set edge voxels to zero
        volume = filter_connected_volume(volume) # connected components analysis
        
        vertices, faces, normals, values = Voxel2SurfMesh(volume, voxelSize=voxelSize, step_size=STEPSIZE)
        
        vertices, faces = smoothSurfMesh(vertices, faces, iterations=args.smooth_iterations)
        vertices, faces = simplifySurfMeshACVD(vertices, faces, target_fraction=0.15)
        
        if not isWatertight(vertices, faces):
            vertices, faces = repairSurfMesh(vertices, faces)    
            
        assert isWatertight(vertices, faces), "surface not watertight after repair"
            
        nodes, elements, tet = Surf2TetMesh(vertices, faces, verbose=0)
        saveTetrahedralMesh(filenameVTK, tet)

        print(f"Finished... {filenameVTK}")
        
    except:

        print(f"Failed...{filenameNRRD}")