#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  3 16:37:21 2022

@author: qian.cao

# Run FEA on Harsha's 700+ new ROIs for the L4 L5 data
# does not perform FEA, meshing only

# commandline tool for meshing a numpy array (isosurface, smoothing, simplification, meshfix)
# this is a simple test code for .._partial.py

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
import glob
import os

if __name__ == "__main__":

    crop_size = 201
    voxel_size = 0.05

    segDir = "/gpfs_projects/qian.cao/data/Segmentations_Otsu"
    segFiles = glob.glob(os.path.join(segDir,"Segmentation-*"))

    outDir = "/gpfs_projects/qian.cao/BoneBox-out/test_20220603_FEA_meshingonly_partial_cl"
    os.makedirs(outDir,exist_ok=True)

    segFile = segFiles[0]
    fhead, ftail = os.path.split(segFile)
    
    vtkFile = os.path.join(outDir,ftail.replace(".nrrd",".vtk"))

    os.system(f"python test_20220603_FEA_meshingonly_partial.py {segFile} {vtkFile} --crop_size {crop_size} --voxel_size {voxel_size}")
