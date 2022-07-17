#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jun  3 16:37:21 2022

Adapted to run on Betsy

@author: qian.cao

# Run FEA on Harsha's 700+ new ROIs for the L4 L5 data
# does not perform FEA, meshing only

# commandline tool for meshing a numpy array (isosurface, smoothing, simplification, meshfix)
# this is a simple test code for .._partial.py

Also, makes config file for betsy jobs

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
import socket
import argparse
import subprocess

if __name__ == "__main__":

    # Determine file to process with arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("job",type=int)
    args =parser.parse_args()
    INDEX = args.job - 1 # NOTE: job indices start at 1, not 0

    print(f"==== {__file__} has startd with argument {args.job} ====")

    NUMFILES = 681 # files for Segmentation Otsu (different for Segmentation L1 Otsu)
    crop_size = 201
    voxel_size = 0.05

    whereami = socket.gethostname()

    if whereami in ["openHPC","didsr-gpu11"]:
        segDir = "/gpfs_projects/qian.cao/data/Segmentations_Otsu"
        outDir = "/gpfs_projects/qian.cao/BoneBox-out/test_20220603_FEA_meshingonly_partial_driver"
    else: # ["betsy01.fda.gov", "bc198.fda.gov", etc]
        segDir = "/scratch/qian.cao/Segmentations_Otsu"
        outDir = "/scratch/qian.cao/BoneBox-out/test_20220603_FEA_meshingonly_partial_driver"

    # Get segmentation files
    segFiles = glob.glob(os.path.join(segDir,"Segmentation-*"))
    segFiles.sort()
    assert len(segFiles)==NUMFILES, "The number of segmentation files is incorrect, the data folder has changed"

    # Make output directory
    os.makedirs(outDir,exist_ok=True)

    segFile = segFiles[INDEX]
    fhead, ftail = os.path.split(segFile)
    
    vtkFile = os.path.join(outDir,ftail.replace(".nrrd",".vtk"))

    print(f"==== running test_20220603_FEA_meshingonly_partial.py {segFile} ====")

    # The csv file includes arguments to this line
    # os.system(f"python ~/BoneBox/examples/test_20220603_FEA_meshingonly_partial.py {segFile} {vtkFile} --crop_size {crop_size} --voxel_size {voxel_size}")
    output = subprocess.check_output(f"python ~/BoneBox/examples/test_20220603_FEA_meshingonly_partial.py {segFile} {vtkFile} --crop_size {crop_size} --voxel_size {voxel_size}", shell=True)
    print(f"check_output: \n{output}")