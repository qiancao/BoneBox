#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 10:10:52 2021

@author: qcao

Analysis for example_topop_tb_v2

"""

import glob
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np

randStates = [1,2,3]
Uls = [0.1,0.2]
Uh = 0.8
Niter = 10
dimXYZ = (100,100,100)

#%% Export images as gifs

for rr in randStates:
    for uu in Uls:
        
        out_dir = "/data/BoneBox-out/topopt/lazy_v2_randstate_" + str(rr) + "_Ul_" + str(uu) + "_Uu_0.8/"
        
        # filepaths
        fp_in = out_dir + "vol_slice50_*.png"
        fp_out = out_dir + "vol_slice50.gif"
        
        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=1000, loop=1)
        
#%% Export BvTv, Elastic Moduli, and Final Phantom

import os

analysis_dir = "/data/BoneBox-out/topopt/analysis_dir/"
if not os.path.exists(analysis_dir):
    os.makedirs(analysis_dir)


Es = np.zeros((len(randStates), len(Uls), Niter))
BvTvs = np.zeros((len(randStates), len(Uls), Niter))

for irr, rr in enumerate(randStates):
    for iuu, uu in enumerate(Uls):
        
        out_dir = "/data/BoneBox-out/topopt/lazy_v2_randstate_" + str(rr) + "_Ul_" + str(uu) + "_Uu_0.8/"
        
        iterVoxelsTotal = np.load(out_dir+"iterVoxelsTotal_10.npy")
        iterElasticModulus = np.load(out_dir+"iterElasticModulus_10.npy")
        
        Es[irr,iuu,:] = iterElasticModulus
        BvTvs[irr,iuu,:] = iterVoxelsTotal / np.prod(dimXYZ)
        
plt.figure()
for irr, rr in enumerate(randStates):
    plt.plot(np.arange(Niter), np.abs(Es[irr,0,:]), "bd--")
    plt.plot(np.arange(Niter), np.abs(Es[irr,1,:]), "ro--")
plt.yscale("log")
plt.xlabel("Iterations")
plt.ylabel("Elastic Modulus (a.u.)")
plt.grid(axis = 'y')
plt.savefig(analysis_dir+"E_vs_iteration.png")
plt.close("all")

plt.figure()
for irr, rr in enumerate(randStates):
    plt.plot(np.arange(Niter), np.abs(BvTvs[irr,0,:]), "bd--")
    plt.plot(np.arange(Niter), np.abs(BvTvs[irr,1,:]), "ro--")
plt.xlabel("Iterations")
plt.ylabel("Bone Volume Fraction (a.u.)")
plt.grid(axis = 'y')
plt.savefig(analysis_dir+"BvTv_vs_iteration.png")
plt.close("all")

#%% Radiomic Features

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

features = np.zeros((len(randStates), len(Uls), 93))

# Define settings for signature calculation
# These are currently set equal to the respective default values
settings = {}
settings['binWidth'] = 25
settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
settings['interpolator'] = sitk.sitkBSpline

# Initialize feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
extractor.enableFeatureClassByName("glcm")

for irr, rr in enumerate(randStates):
    for iuu, uu in enumerate(Uls):
        
        out_dir = "/data/BoneBox-out/topopt/lazy_v2_randstate_" + str(rr) + "_Ul_" + str(uu) + "_Uu_0.8/"
        volume = (np.load(out_dir + "volume_10.npy")*255).astype(int)
        
        volumeSITK = sitk.GetImageFromArray(volume)
        maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
        featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
        featureVectorArray = np.array([featureVector[featureName].item() for featureName in featureVector.keys()])
        features[irr,iuu,:] = featureVectorArray

plt.figure()
for irr, rr in enumerate(randStates):
    plt.plot(np.arange(Niter), np.abs(Es[irr,0,:]), "bd--")
    plt.plot(np.arange(Niter), np.abs(Es[irr,1,:]), "ro--")
    
#%% Visualize Histogram
# out_dir = "/data/BoneBox-out/topopt/lazy_v3_sweep_archive/randstate_2_phantom_ss_2_uu_0/"
# volume0 = np.load(out_dir+"volume_0.npy")
# volume10 = np.load(out_dir+"volume_10.npy")

# fig_vm = np.linspace(0.001,5,1000)
# fig_p = strainEnergy2MaskLazyDiscrete(fig_vm, Ul, Uu)
# # fig_p = strainEnergyDensity2ProbabilityLinear(fig_vm, s0, slope, pmin=-1., pmax=1)

# fig, ax1 = plt.subplots()
# ax1.hist(arrayVM, bins=fig_vm, alpha = 0.5, color = "r")
# ax1.hist(arrayVM, bins=fig_vm, alpha = 0.5, color = "b")
# ax1.set_xlabel('Von Mises Stress (MPa)')
# ax1.set_ylabel('# Elements (Voxels)')

# ax2 = ax1.twinx()
# plt.plot(fig_vm,fig_p,'r--')
# ax2.set_ylabel('Deposition/Resorption Probability')

# plt.savefig(out_dir+"hist_"+str(fea_iter)+".png",bbox_inches='tight')
# plt.close("all")
    
#%% Hexamesh Visualizer
# https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html

nodes = 
elements = 

cpos = [(-22.333459061472976, 23.940062547377757, 1.7396451897739171),
        (-0.04999999999999982, -0.04999999999999982, -0.04999999999999982),
        (0.037118979271661946, -0.040009842455482315, 0.9985095862757241)]

cmap = plt.cm.get_cmap("viridis", 512)
 
# Each cell begins with the number of points in the cell and then the points composing the cell
points = nodes
cells = np.concatenate([(np.ones(elements.shape[0],dtype="int64")*8)[:,None], elements],axis=1).ravel()
celltypes = np.repeat(np.array([vtk.VTK_HEXAHEDRON]), elements.shape[0])
offset = np.arange(elements.shape[0])*9
grid = pv.UnstructuredGrid(offset, cells, celltypes, points)

pl = pv.Plotter(off_screen=True)
pl.add_mesh(grid,show_edges=True, scalars=arrayVM, cmap=cmap, clim=(0,2))
pl.show(window_size=(3000,3000),cpos=cpos,screenshot=out_dir+"volume_"+str(fea_iter)+".png")
    