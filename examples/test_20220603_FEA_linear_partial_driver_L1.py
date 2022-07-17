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

    NUMFILES = 168 # files for Segmentation Otsu (different for Segmentation L1 Otsu)

    whereami = socket.gethostname()

    if whereami in ["openHPC","didsr-gpu11"]:
        inputDir = "/gpfs_projects/qian.cao/BoneBox-out/test_20220603_FEA_meshingonly_partial_driver_L1"
        outDir = "/gpfs_projects/qian.cao/BoneBox-out/test_20220603_FEA_linear_partial_driver_L1"
    else: # ["betsy01.fda.gov", "bc198.fda.gov", etc]
        inputDir = "/scratch/qian.cao/BoneBox-out/test_20220603_FEA_meshingonly_partial_driver_L1"
        outDir = "/scratch/qian.cao/BoneBox-out/test_20220603_FEA_linear_partial_driver_L1"

    # Get segmentation files
    inputFiles = glob.glob(os.path.join(inputDir,"*.vtk"))
    inputFiles.sort()
    assert len(inputFiles)==NUMFILES, "The number of mesh files is incorrect, the data folder has changed"

    # Make output directory
    os.makedirs(outDir,exist_ok=True)

    inputFile = inputFiles[INDEX]
    fhead, ftail = os.path.split(inputFile)
    
    outputFile = os.path.join(outDir,ftail.replace(".vtk",".hdf"))

    print(f"==== running {__file__} for {inputFile} ====")

    # The csv file includes arguments to this line
    # os.system(f"python ~/BoneBox/examples/test_20220603_FEA_meshingonly_partial.py {segFile} {vtkFile} --crop_size {crop_size} --voxel_size {voxel_size}")
    output = subprocess.check_output(f"python ~/BoneBox/examples/test_20220603_FEA_linear_partial.py {inputFile} {outputFile}", shell=True)
    print(f"check_output: \n{output}")