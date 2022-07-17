#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:37:21 2022

@author: qian.cao

# Run FEA on Harsha's 700+ new ROIs for the L4 L5 data
# does not perform FEA, meshing only

# commandline tool for running FEA on with tetrahedral mesh input
# parallel script to be run with .._partial_driver.py

Inputs:

path to input vtk file (with tetrahedral mesh)
path to output hdf file
platten thickness (in mm)


https://stackoverflow.com/questions/11818640/parallel-running-of-several-jobs-in-a-python-script

"""

import numpy as np
import nrrd

import os
import sys

import pyvista as pv
import h5py

# Import BoneBox.FEA
sys.path.append("/home/qian.cao/betsy-setup/chrono_build/build/bin") # TODO: pychrono temporary build
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../bonebox/FEA/"))) # TODO: replace this with relative imports, e.g. from ..FEA.fea import *
from fea import computeFEACompressLinear

# For saving FEA results
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../bonebox/utils/")))
import hdfdict

import argparse

# Set number of MKL cores
import multiprocessing
num_cpus = multiprocessing.cpu_count()
os.environ["MKL_NUM_THREADS"] = str(multiprocessing.cpu_count()-1)

if __name__ == "__main__":

    print(f"======= starting {__file__} with {multiprocessing.cpu_count()-1} cores =====")

    # Parse arguments
    parser = argparse.ArgumentParser(description="Runs uniaxial compression on tetrahedral meshes (.vtk)")
    parser.add_argument("filepath_vtk",type=str,help="string, path to input vtk file")
    parser.add_argument("filepath_hdf",type=str,help="string, path to output hdf file")
    parser.add_argument("--plattenThicknessMM",type=float,default=0.5, help="float, thickness of the compression plates (mm)")
    parser.add_argument("--solver",type=str,default="ParadisoMKL",help="string, solver to use for FEA")
    parser.add_argument("--elasticModulus",type=float,default=17e9, help="float, elastic modulus of the material (Pa)")
    parser.add_argument("--poissonRatio",type=float,default=0.3, help="float, poisson ratio of the materal")
    parser.add_argument("--forceTotal",type=float,default=1., help="float, total force for compression")
    args = parser.parse_args()

    # Input arguments
    filenameVTK = args.filepath_vtk
    filenameHDF = args.filepath_hdf

    plattenThicknessMM = args.plattenThicknessMM
    solver = args.solver
    elasticModulus = args.elasticModulus
    poissonRatio = args.poissonRatio
    forceTotal = args.forceTotal

    # TODO: Remove after debugging
    # plattenThicknessMM = 0.5
    # solver="ParadisoMKL"
    # elasticModulus=17e9
    # poissonRatio=0.3
    # forceTotal=1.
    # filenameVTK = "/scratch/qian.cao/BoneBox-out/test_20220603_FEA_meshingonly_partial_driver/Segmentation-grayscale-2_752-67011-L23.vtk"

    # Load Mesh
    try:
        mesh = pv.read(filenameVTK)
        nodes = np.array(mesh.points) # N by 3 (coordinates in mm)
        elements = mesh.cells.reshape(-1,5)[:,1:] # M by 4

        # Convert coordinates from mm to m (SI units throughout FEA calculation)
        nodes = nodes / 1e3
        plattenThicknessM = plattenThicknessMM / 1e3

    except:
        print("mesh could not be read or incorrect formatting")

    try:
        # FEA
        feaResult = computeFEACompressLinear(nodes, elements, plattenThicknessM, solver=solver, \
            elasticModulus=elasticModulus, poissonRatio=poissonRatio, force_total=forceTotal, verbose=True)

        # Save FEA result to file
        with h5py.File(filenameHDF, "w") as hf:
            hdfdict.dump(feaResult, hf)

        print("FEA finished")

    except:
        print("FEA failed")