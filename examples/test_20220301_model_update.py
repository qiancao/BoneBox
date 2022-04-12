#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:11:02 2021

@author: qcao

sketch of rod plate model and skeleton code for PyChrono

Useful links:
    https://github.com/projectchrono/chrono/blob/03931da6b00cf276fc1c4882f43b2fd8b3c6fa7e/src/chrono/fea/ChElementShellBST.cpp
    https://groups.google.com/g/projectchrono/c/PbTjhc2ek_A (thread on prismatic joints)
    
    
Modified from: test_20220105_rod_plate_models_v4_plates_displacement_vs_thickness.py

Based on test_20200118_Rxyz.py

"""

# the usual suspects
import numpy as np
import numpy.linalg as linalg
import nrrd
import copy

# meshing
from skimage import measure
import tetgen
import trimesh
import pyvista as pv

# finite element library
import pychrono as chrono
import pychrono.fea as fea
import pychrono.pardisomkl as mkl

# SDf and rod-plate FEA routines
import sys
sys.path.append("../bonebox/phantoms/") # from examples folder
sys.path.append("../bonebox/FEA/") # from examples folder

from sdf import *
from feaRP import *
from fea import *
from vtk_utils import *
from TrabeculaeVoronoi import *

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

    # plotter.camera_position = [(-15983.347882469203, -25410.916652156728, 9216.573794734646),
    #  (0.0, 0.0, 0.0),
    #  (0.16876817270434966, 0.24053571467115548, 0.9558555716475535)]
    
    # print(f"saving to f{imgName}...")
    plotter.show(screenshot=f'{imgName}')
    
    # print(plotter.camera_position)
   
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import pyvista as pv
    import vtk
    import os
    
    plt.close("all")
    
    # Output directory
    outDir = "/data/BoneBox-out/test_20220301_model_update/"
    os.makedirs(outDir, exist_ok=True)
    
    # Volume initialization parameters
    volumeExtent = 0.01 # meters = 1 cm
    seedNumPerAxis = 7 # Number
    perturbationRadiusScale = 0.95
    randStates = np.arange(1)
    
    # TODO: These scaling factors will have to be revisited
    plateThicknessScaling = 1e-21
    verticesScaling = 1e6
    
    # Generate Phantoms    
    seedPerturbRadius = volumeExtent/seedNumPerAxis/2 * perturbationRadiusScale # meters = 600 um
    
    for rr in randStates:

        # Generate Voronoi-based skeleton
        Sxyz = (volumeExtent,)*3 # 1cm**3 # (100e-6,100e-6,100e-6)
        Nxyz = (seedNumPerAxis,)*3
        Rxyz = seedPerturbRadius
        v = generateVoronoiSkeleton(Sxyz = np.array(Sxyz), 
                                    Nxyz = Nxyz, randState=rr, 
                                    Rxyz = Rxyz)
        
        # v.vertices = v.vertices * 1e-6
        vertices = v.vertices
        edges = np.array([])
        faces = v.faces # v.faces # v.faces  #v.uniqueFaces # need to break this down to triangles (in v4)
        
        print("Done generating skeleton")
        
        # Faces and vertices augmented for finite-element analysis (apply verticesScaling to finite element analysis)
        verticesFE, facesFE, faceGroup = subdivideFacesWithLocalNeighbors(vertices, faces)
        
        # Boundary conditions
        Lsim = Sxyz[2]*0.6 # Active lengths simulated
        verticesForce = np.nonzero(vertices[:,2] > Lsim/2)[0]
        verticesFixed = np.nonzero(vertices[:,2] < -Lsim/2)[0]
        forceVector = np.array([0,0,-1]).astype(np.double)
        
        # FEA
        thicknesses0, radii0 = 100e-6, 100e-6 # Range of Thickness IS the determining factor of monotonic relationship
        elasticModulus = 17e9 # Pa
        poissonRatio = 0.3
        density = 0 # NO_EFFECT: Density has no impact on results
        barAreas = np.pi*radii0**2*np.ones(len(edges),dtype=float) # m**2 = PI*r**2
        shellThicknesses = thicknesses0*np.ones(len(facesFE),dtype=float) # 
        shellFiberAngles = np.ones(len(facesFE),dtype=float) # NO_EFFECT: Fiber angle has no impact on results
        
        print("Done setting up FEA")
        
        verticesFE1, strainsBar, strainsShell = computeFEARodPlateLinear(verticesFE, edges, facesFE,
                                     verticesForce, verticesFixed, forceVector,
                                     elasticModulus, poissonRatio, density,
                                     barAreas, shellThicknesses, shellFiberAngles,
                                     verticesScaling=verticesScaling,
                                     plateThicknessScaling=plateThicknessScaling)
        
        print("Done FEA")

        # Mean Strain for all faces
        quadrature = lambda q: np.sqrt(q[:,0]**2 + q[:,1]**2+q[:,2]**2)
        strainsShellQuad = quadrature(strainsShell)
        
        strainsShellQuadCombined = []
        faceGroupArr = np.array(faceGroup)
        for g in range(np.max(faceGroup)+1):
            strainsShellQuadCombined.append(np.mean(strainsShellQuad[faceGroupArr == g]))
        strainsShell4Face = np.array(strainsShellQuadCombined)
        
        tanh = lambda x: np.tanh(x)
        faceOpacity = tanh(strainsShell4Face/np.median(strainsShell4Face))
        
        # Render volume
        renderFEA(outDir+f"fea_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}.png", 
                  strainsBar, strainsShell4Face, v, verticesForce, verticesFixed)
        
        displacement = np.mean(linalg.norm(verticesFE1[verticesForce,:] - verticesFE[verticesForce,:],axis=1))
        
        # Random State
        np.save(outDir+f"displacement_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}", displacement)
        np.save(outDir+f"strainsShell4Face_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}", strainsShell4Face)
        
        # Convert to volume
        dims = (200,200,200)
        spacing = np.array(Sxyz) / np.array(dims)
        
        thicknessVoxels = tanh(strainsShellQuad/np.median(strainsShellQuad))*10
        
        # Voxelize
        volume = voxelize_plates(verticesFE, facesFE[:,:3], dims, spacing, thicknessVoxels*5)
        
        volume_platten = addPlatten(volume, 30, plattenValue=np.max(volume), airValue=np.min(volume), trimVoxels=15)
        volume_platten = set_volume_bounds(volume_platten)
        
        vertices_platten, faces_platten, normals, values = Voxel2SurfMesh(volume_platten, voxelSize=spacing, origin=None, level=0, step_size=1, allow_degenerate=False)
        tmesh = trimesh.Trimesh(vertices_platten, faces=faces_platten)
        surf = pv.wrap(tmesh)
        surf.plot()
        
        # plot triplanar
        # plt.imshow(volume_platten[:,:,120].T,cmap='gray',vmin=0,vmax=50);plt.axis("off");plt.show();
        
        volume_platten_gauss = scipy.ndimage.gaussian_filter(volume_platten, 1.5)
        
        plt.imshow(volume_platten_gauss[:,:,120].T,cmap='gray',vmin=0,vmax=30);plt.axis("off");plt.show();
        
        volume_platten_gauss = set_volume_bounds(volume_platten_gauss)
        vertices_platten, faces_platten, normals, values = Voxel2SurfMesh(volume_platten_gauss, voxelSize=spacing, origin=None, level=2, step_size=1, allow_degenerate=False)
        tmesh = trimesh.Trimesh(vertices_platten, faces=faces_platten)
        surf = pv.wrap(tmesh)
        
        surf.plot()
        
        # assert False
    
    # displacements.append(displacementRands)
    # strains.append(strainsRands)
    
    # displacements = np.array(displacements)
    # strains = np.array(strains)
    
    # # E = (F*L0)/(A*dL)
    # E_bone = linalg.norm(forceVector) * Lsim / (Sxyz[0]*Sxyz[1]*displacements) / thicknessScaling
        
    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Elastic Modulus')
    # ax1.boxplot(E_bone.T)
    # plt.xlabel(varName)
    # plt.ylabel("Elastic Modulus (linear, compression)")
    # plt.xticks((np.arange(len(var))+1), (var * volumeExtent/seedNumPerAxis/2))
    # plt.savefig(outDir+f"0_elasticModulus.png")
    
    # fig2, ax2 = plt.subplots()
    # ax2.set_title('Strain')
    # ax2.boxplot(strains.T * thicknessScaling / verticesScaling)
    # plt.xlabel(varName)
    # plt.ylabel("Median Element Strain (linear, compression)")
    # plt.xticks((np.arange(len(var))+1), (var * volumeExtent/seedNumPerAxis/2))
    # plt.savefig(outDir+f"1_meanStrain.png")
    