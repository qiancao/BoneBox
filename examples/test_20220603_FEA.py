#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 16:37:21 2022

@author: qian.cao

# Run FEA on Harsha's 700+ new ROIs

"""

import numpy as np
import matplotlib.pyplot as plt
import glob
import nrrd

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),"../bonebox/FEA/")))
from fea import *

def cropCubeFromCenter(img,length):
    
    x0,y0,z0 = np.array(img.shape)//2
    R = length//2
    
    return img[slice(x0-R,x0+R+1),
               slice(y0-R,y0+R+1),
               slice(z0-R,z0+R+1)]

if __name__ == "__main__":
    
    import h5py
    import hdfdict
    
    # FEA parameters
    voxelSize = (0.05, 0.05, 0.05) # mm
    plattenThicknessVoxels = 10 # voxels
    plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm
    cubeShape = (201, 201, 201)

    # make output directories
    stepsize = 1
    postfix = f"_linear_stepsize_{stepsize}"
    
    outDir = "/gpfs_projects/qian.cao/BoneBox-out/test_20220603_FEA/"
    os.makedirs(outDir,exist_ok=True)
    
    stlDir = outDir+f"stl{postfix}/"
    os.makedirs(stlDir,exist_ok=True)
    
    vtkDir = outDir+f"vtk{postfix}/"
    os.makedirs(vtkDir,exist_ok=True)
    
    hdfDir = outDir+f"fea{postfix}/"
    os.makedirs(hdfDir,exist_ok=True)
    
    # parse segmentation directory
    segDir = "/gpfs_projects/sriharsha.marupudi/ROI_Segmentations_otsu_grayscales/"
    segFiles = glob.glob(segDir+"Segmentation-*")
    
    # list of ROIs to exclude from the analysis
    segExcludeFile = segDir+"Remove_ROIs.txt"
    segExclude = []
    with open(segExcludeFile) as file:
        segExclude = file.readlines()
    segExclude = [x[:-1] for x in segExclude]
    
    def excluded(filepath, segExclude):
        head, tail = os.path.split(filepath)
        return any([x in tail for x in segExclude])
    
    # list of files with successful FEA runs
    segSuccessful = []
    segSuccessfulManifest = outDir+f"fea_manifest{postfix}"
    
    # TODO: for debugging
    import pyvista as pv
    from skimage import measure
    ind = 2
    segFile = segFiles[ind]
    
    #%
    
    # for ind, segFile in enumerate(segFiles):
        
    #     # check if the file is excluded
    #     if not excluded(segFile, segExclude):
            
    #         try:
        
    # Input/Output filenames
    filenameNRRD = segFile
    head, tail = os.path.split(segFile)
    filenameSTL = stlDir+tail+".stl"
    filenameSTLsmooth = stlDir+tail+"_smoothed.stl"
    filenameSTLsimplify = stlDir+tail+"_simplified.stl"
    filenameVTK = vtkDir+tail+".vtk"
    filenameHDF = hdfDir+tail+".hdf5"
    
    print(f"{ind}/{len(segFiles)}: {tail}")
    
    # Elastic Modulus of a real bone ROI
    roiBone, header = nrrd.read(filenameNRRD)
    roiBone = cropCubeFromCenter(roiBone,cubeShape[0]) # crop the ROI to a 1 cm3 volume
    roiBone = addPlatten(roiBone, plattenThicknessVoxels)
    roiBone = set_volume_bounds(roiBone, airValue=None,bounds=2) # set edge voxels to zero
    
    vertices, faces, normals, values = Voxel2SurfMesh(roiBone, voxelSize=voxelSize, step_size=stepsize)
    saveSurfaceMesh(filenameSTL, vertices, faces)
    
    print("Surface Mesh Saved")
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    
    vertices, faces = smoothSurfMesh(vertices, faces)
    saveSurfaceMesh(filenameSTLsmooth, vertices, faces)
    
    print("Surface Mesh Smoothed")
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    
    vertices, faces = simplifySurfMesh(vertices, faces, target_fraction=0.5)
    saveSurfaceMesh(filenameSTLsimplify, vertices, faces)
    
    print("Surface Mesh Simplified")
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    
    #%%
    
    # pv.PolyData(vertices,faces)
    # pv.plot()
    
    # smooth mesh
    
    # simplify mesh
                
    #             nodes, elements, tet = Surf2TetMesh(vertices, faces, verbose=0)
    #             saveTetrahedralMesh(filenameVTK, tet)
    #             feaResult = computeFEACompressLinear(nodes, elements, plattenThicknessMM, solver="ParadisoMKL")
                
    #             with h5py.File(filenameHDF, "w") as hf:
    #                 hdfdict.dump(feaResult, hf)
                    
    #             segSuccessful.append(tail)
                
    #             print("file processed")
            
    #         except:
                
    #             print("file failed")
            
    # # write the successful list to file
    # with open(segSuccessfulManifest,'w') as f:
    #     for item in segSuccessful:
    #         f.write("%s\n" % item)