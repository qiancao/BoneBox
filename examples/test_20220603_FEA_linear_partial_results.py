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
import glob

import pyvista as pv
import h5py

import matplotlib.pyplot as plt

import pandas as pd

# Import BoneBox.FEA
sys.path.append("/home/qian.cao/betsy-setup/chrono_build/build/bin") # TODO: pychrono temporary build
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../bonebox/FEA/"))) # TODO: replace this with relative imports, e.g. from ..FEA.fea import *
from fea import computeFEACompressLinear

# For saving FEA results
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../bonebox/utils/")))
import hdfdict

import argparse

camera_position = [(33.77241683272833, 20.37339381595352, 4.05313061246571),
    (4.9999999813735485, 4.9999999813735485, 4.9999999813735485),
    (0.03299032706089477, -0.000185872956304527, 0.9994556537293985)]

def readHDF(filenameHDF):
    return dict(hdfdict.load(filenameHDF))

def computeStiffness(fea_dict):
    # Computes stiffness from fea results dictionary
    
    d = fea_dict["displacement"][fea_dict["nodeIndA"],:]
    displacement = np.mean(np.sqrt(np.sum(d**2,axis=1)))
    stiffness = fea_dict["force"] / displacement

    return stiffness

def computeMeanVMStress(fea_dict):
    return np.mean(fea_dict["elementVMstresses"])

# def computePistoiaCriterion(fea_dict):

#     eps = fea_dict[]

#     return pistoia

if __name__ == "__main__":

    # Load all data
    fea_paths = ["/scratch/qian.cao/BoneBox-out/test_20220603_FEA_linear_partial_driver/Segmentation*.hdf",
                "/scratch/qian.cao/BoneBox-out/test_20220603_FEA_linear_partial_driver_L1/Segmentation*.hdf"]
    fea_filenames = [] # list of lists (L4L5, L1)
    fea_dicts = []

    for paths in fea_paths:
        fea_filenames.append(glob.glob(paths))

    mesh_paths = ["/scratch/qian.cao/BoneBox-out/test_20220603_FEA_meshingonly_partial_driver/",
            "/scratch/qian.cao/BoneBox-out/test_20220603_FEA_meshingonly_partial_driver_L1/"]

    for ind, filenames in enumerate(fea_filenames):
        dicts = []
        for iind, files in enumerate(filenames):
            
            print(f"{ind}, {iind}")
            fhead,ftail = os.path.split(files)

            d = readHDF(files)
            d["name"] = ftail.replace(".hdf","")

            dicts.append(d)

        fea_dicts.append(dicts)

    # Export ROI stiffness
    dataframeDir = "/scratch/qian.cao/BoneBox-out/dataframes"
    os.makedirs(dataframeDir)
    dataframeNames = ["test_20220603_FEA_linear_partial_driver","test_20220603_FEA_linear_partial_driver_L1"]
    for ind, filenames in enumerate(fea_filenames):
        data = [] # list to store each dataset
        for iind, files in enumerate(filenames):
            fea_dict = fea_dicts[ind][iind]
            data.append((fea_dict["name"], computeStiffness(fea_dict), \
                computeMeanVMStress(fea_dict)))
        df = pd.DataFrame(data, columns=['name', 'stiffness',"mean VM stress"])  
        df.to_pickle(os.path.join(dataframeDir,dataframeNames[ind])+".pkl")


    # # Export PyVista images (this does not work on VS Code without diplay port forwarding)
    # for ind, path in enumerate(fea_paths): # make output directories
    #     fhead, ftail = os.path.split(path)
    #     os.makedirs(os.path.join(fhead,"figures"),exist_ok=True)

    # for ind, filenames in enumerate(fea_filenames):
    #     for iind, files in enumerate(filenames):

    #         print(f"{ind}, {iind}")

    #         fhead, ftail = os.path.split(files)
    #         ftail = ftail.replace(".hdf",".vtk")

    #         mesh = pv.read(os.path.join(mesh_paths[ind],ftail))
            
    #         meshFEA = mesh.copy()
    #         meshFEA.points = mesh.points + fea_dicts[ind][iind]["displacement"]*1e10

    #         filenamePNGfea = os.path.join(fea_paths[ind],"figures",ftail.replace(".vtk",".png"))

    #         pv.set_plot_theme('document')
    #         plotter = pv.Plotter(off_screen=True)
    #         plotter.add_mesh(mesh, color="white",opacity=0.1)
    #         plotter.add_mesh(meshFEA, scalars=fea_dicts[ind][iind]["elementVMstresses"].flatten(),clim=[0,0.15])
    #         plotter.camera_position = camera_position
    #         plotter.show(screenshot=filenamePNGfea)

    # plot von Mises stresses for 1 roi
    # d = dicts[16]
    # plt.figure()
    # plt.hist(d["elementVMstresses"],1000)
    # plt.xlabel("VM Stresses by Element")
    # plt.ylabel("Element Counts")
    # plt.xlim([0,0.5e6])
    # plt.ylim([0,100000])

