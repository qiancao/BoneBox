#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 12:24:50 2022

@author: qcao
"""

# the usual suspects
import numpy as np
import numpy.linalg as linalg
import nrrd
import copy

import matplotlib.pyplot as plt
import pyvista as pv
import vtk
import os

# meshing
from skimage import measure
import tetgen
import trimesh
import pyvista as pv

# finite element library
# import pychrono as chrono
# import pychrono.fea as fea
# import pychrono.pardisomkl as mkl

# SDf and rod-plate FEA routines
import sys
sys.path.append("../bonebox/phantoms/") # from examples folder
sys.path.append("../bonebox/FEA/") # from examples folder

from sdf import *
from feaRP import *
from fea import *
from vtk_utils import *
from TrabeculaeVoronoi import *
from cuSDF import *

import logging
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

def generateVoronoiSkeleton(randState = 123, Sxyz = (100,100,100), Nxyz = (6,6,6), Rxyz = 8.):
    # TODO: to be refactored
    
    mid = lambda Sxyz: tuple(np.array([-np.array(Sxyz)/2, np.array(Sxyz)/2]).T)
    
    # Generate unique faces and edges from Voronoi tessellation
    points = makeSeedPointsCartesian(Sxyz, Nxyz)
    ppoints = perturbSeedPointsCartesianUniformXYZ(points, Rxyz, randState=randState)
    vor, ind = applyVoronoi(ppoints, Sxyz)
    uniqueEdges, uniqueFaces = findUniqueEdgesAndFaces(vor, ind)
    
    # Filter out points, edges and faces outside lim
    indsVertices = filterPointsLimXYZ(vor.vertices, *mid(Sxyz))
    indsEdges = filterByInd(uniqueEdges, indsVertices)
    indsFaces = filterByInd(uniqueFaces, indsVertices)
    
    # Remove vertices, edges and faces outside Sxyz
    vertices, edges, faces = pruneMesh(vor.vertices, uniqueEdges, uniqueFaces)
    
    # Compute edge cosines
    edgeVertices = getEdgeVertices(vertices, edges)
    edgeCosines = computeEdgeCosine(edgeVertices, direction = (0,0,1))
    
    # Compute face properties
    faceVertices = getFaceVertices(vertices, faces)
    faceAreas = computeFaceAreas(faceVertices)
    faceCentroids = computeFaceCentroids(faceVertices)
    faceNormals = computeFaceNormals(faceVertices)
    
    return type("Voronoi",(object,),{"uniqueEdges": uniqueEdges,
                                     "uniqueFaces": uniqueFaces,
                                     "vor": vor,
                                     "indsVertices": indsVertices,
                                     "indsEdges": indsEdges,
                                     "indsFaces": indsFaces,
                                     "edgeVertices": edgeVertices,
                                     "edgeCosines": edgeCosines,
                                     "faceVertices": faceVertices,
                                     "faceAreas": faceAreas,
                                     "faceCentroids": faceCentroids,
                                     "faceNormals": faceNormals,
                                     "vertices": vertices,
                                     "edges": edges,
                                     "faces": faces,
                                     })

def renderFEA(imgName, strainsBar, strainsShell4Face, v, verticesForce, verticesFixed):
    # Plot base structure
    # See: https://docs.pyvista.org/examples/00-load/create-poly.html
    
    pv.set_plot_theme("document")
    
    sigmoid = lambda x: 1/(1 + np.exp(-x))
    tanh = lambda x: np.tanh(x)
    lineWidths = tanh(strainsBar/np.median(strainsBar))*5
    faceOpacity = tanh(strainsShell4Face/np.median(strainsShell4Face))
    
    movingVertices = pv.PolyData(v.vertices[verticesForce,:])
    fixedVertices = pv.PolyData(v.vertices[verticesFixed,:])

    rods = pv.PolyData(v.vertices)
    rods.lines = padPolyData(v.edges)
    
    plates = pv.PolyData(v.vertices)
    plates.faces = padPolyData(v.faces)
    
    plotter = pv.Plotter(shape=(1, 2), border=False, window_size=(2400, 1500), off_screen=True)
    plotter.background_color = 'w'
    plotter.enable_anti_aliasing()
    
    plotter.subplot(0, 0)
    plotter.add_text("Voronoi Skeleton (Rods and Plates)", font_size=24)
    plotter.add_mesh(rods, color='k', show_edges=True)
    plotter.add_mesh(movingVertices,color='r',point_size=10)
    plotter.add_mesh(fixedVertices,color='b',point_size=10)
    
    plotter.subplot(0, 1)
    plotter.add_text("Strain Plates Only", font_size=24)
    plotter.add_mesh(movingVertices,color='r',point_size=10)
    plotter.add_mesh(fixedVertices,color='b',point_size=10)
    
    mesh = pv.PolyData(v.vertices, padPolyData(v.faces))
    # plotter.add_mesh(mesh,scalars=faceOpacity, opacity=faceOpacity, show_scalar_bar=True)
    plotter.add_mesh(mesh,scalars=faceOpacity, opacity=1, show_scalar_bar=True)

    plotter.link_views()

    plotter.camera_position = [(-15983.347882469203, -25410.916652156728, 9216.573794734646),
     (0.0, 0.0, 0.0),
     (0.16876817270434966, 0.24053571467115548, 0.9558555716475535)]
    
    # print(f"saving to f{imgName}...")
    plotter.show(screenshot=f'{imgName}')
    
def set_sdf_boundary(sdf,value=None):
    # expects a unflattened array
    
    if value is None:
        value = np.max(sdf)
    
    sdf[0,:,:] = value
    sdf[-1,:,:] = value
    sdf[:,0,:] = value
    sdf[:,-1,:] = value
    sdf[:,:,0] = value
    sdf[:,:,-1] = value
    
    return sdf

if __name__ == "__main__":
    
    # Output directory
    outDir = "/data/BoneBox-out/test_20220323_Rxyz/"
    os.makedirs(outDir, exist_ok=True)
    
    # Model Inputs
    volume_extent = 0.01 # mm
    
    voronoi_perturbations = [0.0, 0.9]
    voronoi_thicknesses = [200e-6, 130e-6] # mm
    voronoi_nseeds = [6, 6] # number of points
    
    randStates = np.arange(2)
    
    # Voxelization Parameterse
    dims = (200,200,200)
    spacing = np.array(volume_extent) / np.array(dims)
    
    # Results
    all_displacements = []
    all_stiffnesses = []
    all_bvtvs = []
    all_sdfs = []
    all_features = []
    
    # Loop over inputs
    for vv in range(len(voronoi_perturbations)):
        
        displacements = []
        stiffnesses = []
        bvtvs = []
        
        sdfs = [] 
        features = []
        
        for rr, randState in enumerate(randStates):
            
            Sxyz = (volume_extent,)*3
            Nxyz = (voronoi_nseeds[vv],)*3
            Rxyz = volume_extent/voronoi_nseeds[vv]/2 *voronoi_perturbations[vv]
            
            v = generateVoronoiSkeleton(Sxyz = Sxyz, 
                                        Nxyz = Nxyz, randState=randState, 
                                        Rxyz = Rxyz)
            
            vertices = v.vertices
            edges = v.edges
            faces = np.array([]) #v.faces
            
            # Break down faces to triangles
            # verticesFE, facesFE, faceGroup = subdivideFacesWithLocalNeighbors(vertices, faces)
            
            faces_r = (voronoi_thicknesses[vv],)*len(faces)
            edges_r = (voronoi_thicknesses[vv],)*len(edges)
            
            sdf = sdf_rod_plate(vertices, faces, faces_r, edges, edges_r, dims, spacing, k=2)
            
            sdf_platten = addPlatten(sdf, 40, plattenValue=0, airValue=np.max(sdf), trimVoxels=20)
            sdf_platten = set_sdf_boundary(sdf_platten)
            
            # vertices_sdf, faces_sdf, normals, values = Voxel2SurfMesh(sdf_platten, voxelSize=spacing, origin=None, level=0, step_size=1, allow_degenerate=False)
            # tmesh = trimesh.Trimesh(vertices_sdf, faces=faces_sdf)
            # surf = pv.wrap(tmesh)
            
            # pv.set_plot_theme("document")
            # pl = pv.Plotter(window_size=(2400, 1500), off_screen=True)
            # pl.add_mesh(surf)
            # pl.camera_position = [(0.0280644095056988, 0.020993754093686964, 0.01066078508011021),
            #  (0.0049749998725019395, 0.0049749998725019395, 0.004974999972546357),
            #  (-0.17396743835431502, -0.09707424333477963, 0.9799550610479123)]
            # pl.show(screenshot=outDir+f'{vv}_{rr}')
            
            # nodes, elements, tet = Surf2TetMesh(vertices_sdf, faces_sdf, order=1, verbose=1)
            
            # feaResult = computeFEACompressLinear(nodes, elements, 0.002, \
            #                                      elasticModulus=17e9, poissonRatio=0.3, \
            #                                          force_total = 1, solver="ParadisoMKL")
            
            # Define settings for signature calculation
            # These are currently set equal to the respective default values
            settings = {}
            settings['binWidth'] = 25
            settings['resampledPixelSpacing'] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
            settings['interpolator'] = sitk.sitkBSpline
            
            # Initialize feature extractor
            extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
            extractor.enableFeatureClassByName("glcm")
            
            volume = (sdf<=0).astype(int)*255
            volumeSITK = sitk.GetImageFromArray(volume)
            maskSITK = sitk.GetImageFromArray(np.ones(volume.shape).astype(int))
            featureVector = extractor.computeFeatures(volumeSITK, maskSITK, imageTypeName="original")
            
            featureNames = featureVector.keys()
            featureValues = [float(featureVector[featureName].item()) for featureName in featureVector.keys()]
            
            print(len(vertices))
            print(len(edges))
                
            bvtvs.append(np.sum(sdf<=0)/sdf.size)
            # stiffnesses.append(computeFEAElasticModulus(feaResult))
            sdfs.append(sdf)
            features.append(featureValues)
            
        all_bvtvs.append(bvtvs)
        all_stiffnesses.append(stiffnesses)
        all_sdfs.append(sdfs)
        all_features.append(features)
        
    all_features = np.array(all_features)
    
    plt.figure()
    plt.plot(all_features[0,0,:],'b-')
    plt.plot(all_features[1,0,:],'r-')
    plt.plot(all_features[0,1,:],'g-')
    plt.plot(all_features[1,1,:],'m-')
    plt.yscale("log")