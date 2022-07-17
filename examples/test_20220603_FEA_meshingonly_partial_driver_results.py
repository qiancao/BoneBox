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

    # check log files to see if any meshes failed mid-run
    sysoutdir = "/home/qian.cao/sysout"

    NUMFILES = 681 # files for Segmentation Otsu (different for Segmentation L1 Otsu)

    inds_finished = []
    for ind in range(NUMFILES):

        logfile = glob.glob(os.path.join(sysoutdir,f"MESHING*.{ind+1}"))[0]

        with open(logfile) as file:
            contents = file.read()
            if "Finished" in contents:
                inds_finished.append(ind+1)

    set(np.arange(681)+1) - set(inds_finished) # 38, 153 failed