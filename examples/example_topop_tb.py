#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 02:58:30 2021

@author: qcao
"""

import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.phantoms.TrabeculaeVoronoi import *
from bonebox.FEA.fea import *

import numpy as np
import matplotlib.pyplot as plt

from skimage.morphology import ball, closing, binary_dilation

def volume2mesh(volume, dimXYZ, voxelSize):
    """
    Returns node (in abs coordinates) and elements corresponding to volume.
    """
    nodes, elements = Voxel2HexaMeshIndexCoord(volume)
    nodes = Index2AbsCoords(nodes, volumeSizeVoxels=dimXYZ, voxelSize=voxelSize)
    return nodes, elements

def performFEA(nodes, elements, plateThicknessMM, elasticModulus, poissonRatio):
    """
    Runs FEA on nodes and elements.
    """
    feaResult = computeFEACompressLinearHex(nodes, elements, plateThickness=plateThickness, \
                             elasticModulus=elasticModulus, poissonRatio=poissonRatio, \
                             force_total = 1, solver="ParadisoMKL")
    return feaResult

def extractStresses(feaResult, dimXYZ, voxelSize):
    """
    Runs FEA on nodes and elements.
    """
    # Take absolute stresses
    arrayVM = feaResult["elementVMstresses"].flatten()
    arrayXX = np.abs(feaResult["elementStresses"][:,0])
    arrayYY = np.abs(feaResult["elementStresses"][:,1])
    arrayZZ = np.abs(feaResult["elementStresses"][:,2])
    
    # Convert to indexing coordinates, assign to voxels
    nodesIndexCoord = Abs2IndexCoords(nodes, dimXYZ, voxelSize=voxelSize, origin=(0,0,0))
    volumeVM = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayVM)
    volumeXX = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayXX)
    volumeYY = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayYY)
    volumeZZ = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayZZ)
        
    xyz = nodes[elements,:][:,0,:] + 0.5
    
    return xyz, arrayVM, arrayXX, arrayYY, arrayZZ, volumeVM, volumeXX, volumeYY, volumeZZ

# Volume update
def setVoxelsOne(volume, xyz):
    volume[tuple(xyz.T)] = 1
    return volume
    
def setVoxelsZero(volume, xyz):
    volume[tuple(xyz.T)] = 0
    return volume

def setNanToZero(x):
    x[np.isnan(x)] = 0
    return x

def stressStrainVolume2StrainEnergyDensity(stress, strain, volume):
    """    
    For linear isotropic materials undergoing small strains.

    stress/strain (Nelements x 6) where for each row components are [xx, yy, zz, xy, yz, xz]
    
    """
    
    strainEnergyDensity = 0.5 * np.sum(stress[:,:3] * strain[:,:3],axis=1) \
        + np.sum(stress[:,3:] * strain[:,3:],axis=1)
    
    return strainEnergyDensity

def strainEnergyDensity2ProbabilityLinear(strainEnergyDensity, s0, slope, pmin=-1., pmax=1):
    """
    Computes the probability of bone removal/addition given its strain energy, based on lazy model
    
    Christen et al. Bone remodelling in humans is load-driven but not lazy
    
    stress: np.array of voxel stress values
    s0 (scalar): stress level corresponding to homeostasis
    slope (scalar): slope of increasing probability of addition wrt stress
    pmin, pmax: minimum and maximum probability of voxel removal/addition
    
    """
    
    prop = slope * (strainEnergyDensity - s0)
    
    return np.clip(prob, -1, 1)

def strainEnergy2ProbabilityLazy(strainEnergyDensity, Ul, Uu, Ce):
    """
    Computes probability of bone removal/addition given its strain energy, based on lazy model.

    Nowak et al. New aspects of the trabecular bone remodeling regulatory model resulting from shape optimization studies.
    
    Note that output is not Elastic modulus but a probability of voxel undergoing remodeling.
    """
    
    prob = np.zeros(strainEnergyDensity.shape)
    prob[strainEnergyDensity>Uu] = Ce*(strainEnergyDensity - Uu)
    prob[strainEnergyDensity>Ul] = Ce*(strainEnergyDensity - Ul)
    
    return np.clip(prob, -1, 1)

def sampleProbabilityAddRemove(prob, randState):
    """
    Generates output array with +1 denoting addition, -1 denoting removal, and 0 denoting no change,
    given probabilities defined in prob.

    """
    
    # Generate uniform distribution in (-1,1)
    mask = sampleUniformZeroOne(prob.shape, randState=randState) * 2 - 1
    
    mask[(prob>0)&(mask>prob)] = 1
    mask[(prob<0)&(mask<prob)] = -1
    mask[(mask!=-1)&(mask!=1)] = 0
    
    return mask

def volumeVoxelGrowRemove(volume, xyz, mask):
    """
    Add (8-neighborhood) voxels in a volume according to mask
    Remove 
    
    Parameters
    ----------
    volume : Integer array of 0 and 1's
        Initial Volume.
    xyz : tuple of 3 arrays
        tuple denoting coordinates cooresponding to mask
    mask : 
        Voxels to add (+1) and voxel to remove (-1)
        

    Returns
    -------
    volume.

    """
    
    # Neighborhood (6-connected 3D ball)
    nbh = ball(1)
    
    # Convert voxel coordinates to volume (TODO: not needed, refactor later)
    dvolume = np.zeros(volume.shape, dtype=int)
    dvolume[xyz] = mask
    
    # Assign voxels to be added
    volumeAdd = (dvolume==1)
    volumeAdd = binary_dilate(volumeAdd,nbh)
    volume[volumeAdd==1] = 1
    
    # Assign voxels to be removed
    volumeRemove = (dvolume==-1)
    volume[volumeRemove==1] = 0
    
    return volume

if __name__ == "__main__":
    
    import pyvista as pv
    pv.set_plot_theme("document")
    
    #% Make base structure
    
    # Parameters for generating phantom mesh
    Sxyz, Nxyz = (10,10,15), (8,8,8) # volume extent in XYZ (mm), number of seeds along XYZ
    Rxyz = 0.5
    # edgesRetainFraction = 0.5
    # facesRetainFraction = 0.1
    dilationRadius = 3 # (voxels)
    randState = 123 # for repeatability
    
    morphClosingMask = np.ones((3,3,3)) # mask for morphological closing
    
    # Parameters for generating phantom volume
    volumeSizeVoxelsInitial = (100,100,150)
    voxelSize = np.array(Sxyz) / np.array(volumeSizeVoxelsInitial)
    
    # FEA parameters
    plattenThicknessVoxels = 2 # voxels
    elasticModulus = 17e9
    poissonRatio = 0.3
    forceTotal = 1
    
    # Generate faces and edges
    points = makeSeedPointsCartesian(Sxyz, Nxyz)
    ppoints = perturbSeedPointsCartesianUniformXYZ(points, Rxyz, randState=randState)
    vor, ind = applyVoronoi(ppoints, Sxyz)
    uniqueEdges, uniqueFaces = findUniqueEdgesAndFaces(vor, ind)
    
    # Compute edge cosines
    edgeVertices = getEdgeVertices(vor.vertices, uniqueEdges)
    edgeCosines = computeEdgeCosine(edgeVertices, direction = (0,0,1))
    
    # Compute face properties
    faceVertices = getFaceVertices(vor.vertices, uniqueFaces)
    faceAreas = computeFaceAreas(faceVertices)
    faceCentroids = computeFaceCentroids(faceVertices)
    faceNormas = computeFaceNormals(faceVertices)
    
    # Make edge volume
    # uniqueEdgesRetain = uniqueEdges
    uniqueFacesRetain = uniqueFaces
    # volumeEdges = makeSkeletonVolumeEdges(vor.vertices, uniqueEdgesRetain, voxelSize, volumeSizeVoxels)
    volumeFaces = makeSkeletonVolumeFaces(vor.vertices, uniqueFacesRetain, voxelSize, volumeSizeVoxelsInitial)
    
    # Morphological closing on volume of edges
    # volumeSkeleton = closing(np.logical_or(volumeEdges,volumeFaces),morphClosingMask)
    # volumeSkeleton = closing(volumeEdges, morphClosingMask)
    volume = volumeFaces[:,:,25:125]
    volumeSizeVoxels = volume.shape

    # volume = dilateVolumeSphereUniform(volumeSkeleton, dilationRadius)
    
    import os
    
    out_dir = "/data/BoneBox-out/topopt/R_"+str(Rxyz)+"/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    #%% Iteratively perform FEA and augment volume shape
    
    for fea_iter in np.arange(5):
    
        # Convert to hex mesh for FEA
        volume[:,:,0] = 0
        volume[:,:,-1] = 0
        volumePlatten = addPlatten(volume, plattenThicknessVoxels)
        nodes, elements = Voxel2HexaMeshIndexCoord(volumePlatten)
        nodes = Index2AbsCoords(nodes, volumeSizeVoxels, voxelSize=voxelSize)
        
        
        
        feaResult = computeFEACompressLinearHex(nodes, elements, plateThickness=plattenThicknessVoxels * voxelSize[0], \
                                 elasticModulus=elasticModulus, poissonRatio=poissonRatio, \
                                 force_total = forceTotal, solver="ParadisoMKL")
        
        elasticModulus = computeFEAElasticModulus(feaResult)
    
        # Take absolute stresses
        arrayVM = setNanToZero(feaResult["elementVMstresses"].flatten())
        arrayXX = setNanToZero(np.abs(feaResult["elementStresses"][:,0]))
        arrayYY = setNanToZero(np.abs(feaResult["elementStresses"][:,1]))
        arrayZZ = setNanToZero(np.abs(feaResult["elementStresses"][:,2]))
        
        # Convert to indexing coordinates, assign to voxels
        nodesIndexCoord = Abs2IndexCoords(nodes, volumeSizeVoxels, voxelSize=voxelSize, origin=(0,0,0))
        volumeVM = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayVM)
        volumeXX = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayXX)
        volumeYY = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayYY)
        volumeZZ = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayZZ)
            
        xyz = (nodesIndexCoord[elements,:][:,0,:] + 0.5).astype(int)
        
        # Volume Update
        t0percentile = 20
        t0mask = arrayVM < np.percentile(arrayVM,t0percentile)
        
        dvolume = setVoxelsOne(np.zeros(volume.shape),xyz[t0mask,:])
        
        np.save(out_dir+"volume_"+str(fea_iter),volume)
        np.save(out_dir+"dvolume_"+str(fea_iter),dvolume)
        
        #%%
        
        plt.imshow(volume[:,50,:].T + dvolume[:,50,:].T)
        plt.axis("off")
        plt.savefig(out_dir+"vol_slice50_"+str(fea_iter)+".png")
        plt.close("all")
        
        volume = ((volume - dvolume)>0).astype(float)
        
        #%% Visualize stress histogram
        
        fig, axs = plt.subplots(2,2,figsize=(20,12)); axs = axs.ravel()
        fig.suptitle('Stress Components Histogram')
        axs[0].hist(arrayVM,bins=np.linspace(0,3*np.median(arrayVM),200)); axs[0].set_ylabel("VM Stress")
        axs[1].hist(arrayXX,bins=np.linspace(0,3*np.median(arrayXX),200)); axs[1].set_ylabel("XX Stress")
        axs[2].hist(arrayYY,bins=np.linspace(0,3*np.median(arrayYY),200)); axs[2].set_ylabel("YY Stress")
        axs[3].hist(arrayZZ,bins=np.linspace(0,3*np.median(arrayZZ),200)); axs[3].set_ylabel("ZZ Stress")
        plt.savefig(out_dir+"hist_"+str(fea_iter)+".png")
        plt.close("all")
            
        #%% Hexamesh Visualizer
        # https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html
        
        dnodes = feaResult['displacement']
        
        cpos = [(-22.333459061472976, 23.940062547377757, 1.7396451897739171),
                (-0.04999999999999982, -0.04999999999999982, -0.04999999999999982),
                (0.037118979271661946, -0.040009842455482315, 0.9985095862757241)]
        
        import vtk
        from pyvistaqt import BackgroundPlotter
        
        cmap = plt.cm.get_cmap("viridis", 512)
     
        # Each cell begins with the number of points in the cell and then the points
        # composing the cell
        points = nodes
        cells = np.concatenate([(np.ones(elements.shape[0],dtype="int64")*8)[:,None], elements],axis=1).ravel()
        celltypes = np.repeat(np.array([vtk.VTK_HEXAHEDRON]), elements.shape[0])
        offset = np.arange(elements.shape[0])*9
        grid = pv.UnstructuredGrid(offset, cells, celltypes, points)
        
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(grid,show_edges=True, scalars=arrayVM, cmap=cmap, clim=(0,0.3))
        # pl.camera.azimuth = 0
        pl.show(window_size=(3000,3000),cpos=cpos,screenshot=out_dir+"volume_"+str(fea_iter)+".png")
        
        #%% Hexamesh Visualizer
        # https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html
        
        dnodes = feaResult['displacement']
        
        cpos = [(-22.333459061472976, 23.940062547377757, 1.7396451897739171),
                (-0.04999999999999982, -0.04999999999999982, -0.04999999999999982),
                (0.037118979271661946, -0.040009842455482315, 0.9985095862757241)]
        
        import vtk
        from pyvistaqt import BackgroundPlotter
        
        cmap = plt.cm.get_cmap("viridis", 512)
     
        # Each cell begins with the number of points in the cell and then the points
        # composing the cell
        points = nodes + dnodes
        cells = np.concatenate([(np.ones(elements.shape[0],dtype="int64")*8)[:,None], elements],axis=1).ravel()
        celltypes = np.repeat(np.array([vtk.VTK_HEXAHEDRON]), elements.shape[0])
        offset = np.arange(elements.shape[0])*9
        grid = pv.UnstructuredGrid(offset, cells, celltypes, points)
        
        pl = pv.Plotter(off_screen=True)
        pl.add_mesh(grid,show_edges=True, scalars=arrayVM, cmap=cmap, clim=(0,0.3))
        # pl.camera.azimuth = 0
        pl.show(window_size=(3000,3000),cpos=cpos,screenshot=out_dir+"dvolume_"+str(fea_iter)+".png")
    
        
        # pl.show()
        
        
        # grid.plot(show_edges=True, scalars=arrayVM, cmap=cmap, clim=(0,0.3), cpos = "xz",
        #                full_screen=True, screenshot=out_dir+"volume_"+str(fea_iter)+".png", off_screen=True)
    
    # plotter = BackgroundPlotter(window_size=(3000,3000))
    # plotter.add_mesh(grid,show_edges=True, scalars=arrayVM, cmap=cmap, clim=(0,0.3))
    # plotter.close()
    
    # pl = grid.plot(show_edges=True, scalars=arrayVM, cmap=cmap, clim=(0,0.3),
    #               full_screen=True, off_screen=True)
    # pl.camera.azimuth = 45
    # pl.show()
    
    
    # plotter.show(screenshot=out_dir+'volume_0.png')

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(xyz[:,0], xyz[:,1], xyz[:,2], \
    #              c=feaResult["elementVMstresses"].flatten(), cmap='plasma');
    
    # ma = (nodesIndexCoord+0.5).astype(int)
        
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # # ax.set_aspect('equal')
    # ax.voxels(volume, edgecolor="k")
    # plt.show()