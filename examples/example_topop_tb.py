#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 02:58:30 2021

@author: qcao
"""

import os
import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.phantoms.TrabeculaeVoronoi import *
from bonebox.FEA.fea import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

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
    
    Helpful link:
        http://homepages.engineering.auckland.ac.nz/~pkel015/SolidMechanicsBooks/Part_I/BookSM_Part_I/08_Energy/08_Energy_02_Elastic_Strain_Energy.pdf
    
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
    
    prob = slope * (strainEnergyDensity - s0)
    
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
    
    mask[(prob>0)&(mask<prob)] = 1
    mask[(prob<0)&(mask>prob)] = -1
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
    # nbh = ball(1)
    nbh = np.ones((3,3,3))
    
    # Convert voxel coordinates to volume (TODO: not needed, refactor later)
    dvolume = np.zeros(volume.shape, dtype=int)
    dvolume[xyz] = mask
    
    # Assign voxels to be added
    volumeAdd = (dvolume==1)
    volumeAdd = binary_dilation(volumeAdd,nbh)
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
    Sxyz, Nxyz = (5,5,7.5), (5,5,7) # volume extent in XYZ (mm), number of seeds along XYZ
    Rxyz = 0.5
    # edgesRetainFraction = 0.5
    # facesRetainFraction = 0.1
    dilationRadius = 3 # (voxels)
    randState = 1 # for repeatability
    
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
    uniqueFacesRetain = uniqueFaces
    volumeFaces = makeSkeletonVolumeFaces(vor.vertices, uniqueFacesRetain, voxelSize, volumeSizeVoxelsInitial)
    
    # Morphological closing on volume of edges
    volume = volumeFaces[:,:,25:125]
    volumeSizeVoxels = volume.shape
    
    # General optimization parameters
    Niters = 50
    
    # Run Indicators
    iterVoxelsChanged = np.zeros(Niters)
    iterVoxelsTotal = np.zeros(Niters)
    
    # Adaptation parameters, assume intercept of -1 for 0 stress
    # using VM stress instead
    s0 = 0.2
    pf = np.polyfit([0,s0],[-1,0],1)
    k = 1
    slope = pf[0] * k
    
    # Make output directory
    out_dir = "/data/BoneBox-out/topopt/randstate_"+str(randState)+"_s0_"+str(s0)+"_k_"+str(k)+"/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    #%% Iteratively perform FEA and augment volume shape
    
    np.save(out_dir+"volume_"+str(0), volume)
    
    for fea_iter in np.arange(Niters):
        
        # Save original volume at the start of iteration
        volume0 = volume.copy()
    
        # Convert to hex mesh for FEA
        volume[:,:,0] = 0
        volume[:,:,-1] = 0
        volumePlatten = addPlatten(volume, plattenThicknessVoxels)
        
        nodes, elements = Voxel2HexaMeshIndexCoord(volumePlatten)
        nodes = Index2AbsCoords(nodes, volumeSizeVoxels, voxelSize=voxelSize)
        
        # Finite element analysis
        feaResult = computeFEACompressLinearHex(nodes, elements, plateThickness=plattenThicknessVoxels * voxelSize[0], \
                                 elasticModulus=elasticModulus, poissonRatio=poissonRatio, \
                                 force_total = forceTotal, solver="ParadisoMKL")
        
        # Compute elastic modulus
        elasticModulus = computeFEAElasticModulus(feaResult)
        
        # Index coordinates of elements (voxel centers)
        nodesIndexCoord = Abs2IndexCoords(nodes, volumeSizeVoxels, voxelSize=voxelSize, origin=(0,0,0))
        xyz = (nodesIndexCoord[elements,:][:,0,:] + 0.5).astype(int)
        
        np.save(out_dir+"xyz_"+str(fea_iter), xyz)
        
        xyz = tuple(xyz.T)
        
        # Take absolute stresses
        arrayVM = setNanToZero(feaResult["elementVMstresses"].flatten())
        arrayXX = setNanToZero(np.abs(feaResult["elementStresses"][:,0]))
        arrayYY = setNanToZero(np.abs(feaResult["elementStresses"][:,1]))
        arrayZZ = setNanToZero(np.abs(feaResult["elementStresses"][:,2]))
        
        np.save(out_dir+"arrayVM_"+str(fea_iter), arrayVM)
        
        # Compute strain energy, probability, 
        # strainEnergyDensity = setNanToZero(stressStrainVolume2StrainEnergyDensity(feaResult["elementStresses"], \
        #                                                                           feaResult["elementStrains"], \
        #                                                                               np.prod(voxelSize)))
        # prob = strainEnergyDensity2ProbabilityLinear(strainEnergyDensity, s0, slope, pmin=-1., pmax=1)
        
        prob = strainEnergyDensity2ProbabilityLinear(arrayVM, s0, slope, pmin=-1., pmax=1)
        mask = sampleProbabilityAddRemove(prob, randState = randState + fea_iter)
        volume1 = volumeVoxelGrowRemove(volume, xyz, mask)
        volume1[:,:,0] = 0
        volume1[:,:,-1] = 0

        # Convert to indexing coordinates, assign to voxels
        volumeVM = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayVM)
        volumeXX = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayXX)
        volumeYY = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayYY)
        volumeZZ = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, arrayZZ)
        # volumeSED = HexaMeshIndexCoord2VoxelValue(nodesIndexCoord, elements, volumeSizeVoxels, strainEnergyDensity)

        np.save(out_dir+"volume_"+str(fea_iter+1), volume1)
        
        # print("median strain energy:"+str(np.median(strainEnergyDensity)))
        print("median VM Stress:"+str(np.median(arrayVM)))

        #%% Show volume slice (sagittal)
        
        dvolume = volume1.astype(int)-volume0.astype(int)
        volumeShow = np.zeros(volume.shape).astype(int)
        volumeShow[volume0==1] = 2
        volumeShow[dvolume==-1] = 1
        volumeShow[dvolume==1] = 3
        
        # make a color map of fixed colors
        cmap = colors.ListedColormap(['white','red','gray','blue'])
        bounds=[-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        
        fig = plt.figure(frameon=False)
        plt.imshow(volumeShow[:,50,:].T, cmap=cmap, norm=norm)
        plt.axis("off")
        plt.axis("tight")
        plt.savefig(out_dir+"vol_slice50_"+str(fea_iter)+".png")
        plt.close("all")
        
        #%% Show mask histogram
        
        plt.hist(mask,10)
        plt.savefig(out_dir+"dvolhist_slice50_"+str(fea_iter)+".png")
        plt.close("all")
        
        #%% Visualize stress histogram
        
        fig_vm = np.linspace(0.001,5,1000)
        fig_p = strainEnergyDensity2ProbabilityLinear(fig_vm, s0, slope, pmin=-1., pmax=1)
        
        fig, ax1 = plt.subplots()
        ax1.hist(arrayVM,bins=fig_vm)
        ax1.set_xlabel('Von Mises Stress (MPa)')
        ax1.set_ylabel('# Elements (Voxels)')
        
        ax2 = ax1.twinx()
        plt.plot(fig_vm,fig_p,'r--')
        ax2.set_ylabel('Deposition/Resorption Probability')
        
        plt.savefig(out_dir+"hist_"+str(fea_iter)+".png")
        plt.close("all")
            
        #%% Hexamesh Visualizer
        # https://docs.pyvista.org/examples/00-load/create-unstructured-surface.html
        
        cpos = [(-22.333459061472976, 23.940062547377757, 1.7396451897739171),
                (-0.04999999999999982, -0.04999999999999982, -0.04999999999999982),
                (0.037118979271661946, -0.040009842455482315, 0.9985095862757241)]
        
        import vtk
        from pyvistaqt import BackgroundPlotter
        
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
        
        #%%
        
        # iterVoxelChbanged and Total
        iterVoxelsChanged[fea_iter] = np.sum(np.abs(dvolume))
        iterVoxelsTotal[fea_iter] = np.sum(volume1)
        
        fig, ax1 = plt.subplots()
        ax1.plot(np.arange(Niters),iterVoxelsTotal,'k--')
        ax1.set_xlabel('Iteration (#)')
        ax1.set_ylabel('Total Bone Voxels')
        
        ax2 = ax1.twinx()
        plt.plot(np.arange(Niters),iterVoxelsChanged,'k-')
        ax2.set_ylabel('Bone Voxel Change')
        
        plt.savefig(out_dir+"iters.png")
        plt.close("all")
        
        np.save(out_dir+"iterVoxelsChanged_"+str(fea_iter+1), iterVoxelsChanged)
        np.save(out_dir+"iterVoxelsTotal_"+str(fea_iter+1), iterVoxelsTotal)
        