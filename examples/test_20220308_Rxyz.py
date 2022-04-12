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

from feaRP import *
from fea import *
from vtk_utils import *
from TrabeculaeVoronoi import *

from cuSDF import *

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
    # plotter.show()

    plotter.camera_position = [(0.027319263101599902, -0.026339552043427594, 0.013127793077424376),
     (-4.738350340152816e-05, -1.0006934576238784e-05, 6.728825879808234e-05),
     (-0.23312363115051704, 0.22676760723875422, 0.9456372586284911)]
    
    # print(f"saving to f{imgName}...")
    plotter.show(screenshot=f'{imgName}')
    
    # print(plotter.camera_position)
   
def plates2image(vertices, faces, thicknesses, dims, spacing):
    """

    thicknesses: radii of plates in voxels, same len as faces
    dims: number of voxels in xyz
    spacing: voxel size
    
    """

    image = np.zeros(dims, dtype=np.uint8)
    
    for ind, face in enumerate(faces):
        
        print(f"{ind}/{len(faces)}")
        
        grid = pv.UniformGrid(
            dims = np.array(dims),
            spacing = np.array(spacing),
            origin = -(np.array(dims)*np.array(spacing))/2,
        )
        
        mesh = pv.PolyData(vertices, padPolyData([face]))
        
        grid_dist = grid.compute_implicit_distance(mesh)
        
        dist = grid_dist.point_data['implicit_distance']
        dist = np.array(dist).reshape(dims)
        
        dist = (np.abs(dist) < thicknesses[ind]).astype(np.uint8)
    
        image = np.maximum(image, dist)
        
    return image
   
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import pyvista as pv
    import vtk
    import os
    
    # Output directory
    outDir = "/data/BoneBox-out/test_20220309_Rxyz/"
    os.makedirs(outDir, exist_ok=True)
    
    #%%
    
    # TODO: These scaling factors will have to be revisited
    thicknessScaling = 1
    verticesScaling = 1
    
    # interesting, monotonic relationship only observed smaller than 1e-8
    var = np.linspace(0.8, 1, 1)
    varName = "Perturbation Radius (um)"
    
    # random states
    # rands = np.arange(3)
    rands = np.arange(1)
    
    # save displacement and strains
    displacements = []
    strains = []
    
    # Volume initialization parameters
    volumeExtent = 0.01 # meters = 1 cm
    seedNumPerAxis = 6 # Number
    
    for vv in var:
        
        displacementRands = []
        strainsRands = []
        
        seedPerturbRadius = volumeExtent/seedNumPerAxis/2 * vv # meters = 600 um
        
        for rr in rands:
        
            print(vv)
        
            plt.close("all")
            
            # Generate Voronoi-based skeleton
            Sxyz = (volumeExtent,)*3 # 1cm**3 # (100e-6,100e-6,100e-6)
            Nxyz = (seedNumPerAxis,)*3
            Rxyz = seedPerturbRadius
            v = generateVoronoiSkeleton(Sxyz = np.array(Sxyz)*verticesScaling, 
                                        Nxyz = Nxyz, randState=rr, 
                                        Rxyz = Rxyz*verticesScaling)
            
            # v.vertices = v.vertices * 1e-6
            vertices = v.vertices
            edges = np.array([])
            faces = v.faces # v.faces # v.faces  #v.uniqueFaces # need to break this down to triangles (in v4)
            
            print("Done generating skeleton")
            
            # Faces and vertices augmented for finite-element analysis
            verticesFE, facesFE, faceGroup = subdivideFacesWithLocalNeighbors(vertices, faces)
            
            # Boundary conditions
            Lsim = Sxyz[2]*0.6 # Active lengths simulated
            verticesForce = np.nonzero(vertices[:,2] > Lsim/2*verticesScaling)[0]
            verticesFixed = np.nonzero(vertices[:,2] < -Lsim/2*verticesScaling)[0]
            forceVector = np.array([0,0,-1]).astype(np.double)
            
            # FEA
            thicknesses0, radii0 = 50e-6*thicknessScaling, 100e-6 # Range of Thickness IS the determining factor of monotonic relationship
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
                                         barAreas, shellThicknesses, shellFiberAngles)
            
            print("Done FEA")
            
            # assert False
            
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
            renderFEA(outDir+f"fea_{vv}_{rr}.png", strainsBar, strainsShell4Face, v, verticesForce, verticesFixed)
            
            displacement = np.mean(linalg.norm(verticesFE1[verticesForce,:] - verticesFE[verticesForce,:],axis=1))
            
            # Random State
            displacementRands.append(displacement)
            strainsRands.append(np.median(strainsShell4Face))
            
            # Convert to volume
            dims = (200,200,200)
            spacing = np.array(Sxyz) / np.array(dims)
            
            sdf = sdf_rod_plate(verticesFE, facesFE[:,:3], shellThicknesses, edges, [], dims, spacing, k=2)
            
            sdf_platten = addPlatten(sdf, 30, plattenValue=0, airValue=np.max(sdf), trimVoxels=10)
            sdf_platten = set_sdf_boundary(sdf_platten)
            vertices_sdf, faces_sdf, normals, values = Voxel2SurfMesh(sdf_platten, voxelSize=spacing, origin=None, level=0, step_size=1, allow_degenerate=False)
            tmesh = trimesh.Trimesh(vertices_sdf, faces=faces_sdf)
            surf = pv.wrap(tmesh)
            surf.plot()
            
            assert False
        
        displacements.append(displacementRands)
        strains.append(strainsRands)
    
    displacements = np.array(displacements)
    strains = np.array(strains)
    
    # E = (F*L0)/(A*dL)
    E_bone = linalg.norm(forceVector) * Lsim / (Sxyz[0]*Sxyz[1]*displacements) / thicknessScaling
        
    fig1, ax1 = plt.subplots()
    ax1.set_title('Elastic Modulus')
    ax1.boxplot(E_bone.T)
    plt.xlabel(varName)
    plt.ylabel("Elastic Modulus (linear, compression)")
    plt.xticks((np.arange(len(var))+1), (var * volumeExtent/seedNumPerAxis/2))
    plt.savefig(outDir+f"0_elasticModulus.png")
    
    fig2, ax2 = plt.subplots()
    ax2.set_title('Strain')
    ax2.boxplot(strains.T * thicknessScaling / verticesScaling)
    plt.xlabel(varName)
    plt.ylabel("Median Element Strain (linear, compression)")
    plt.xticks((np.arange(len(var))+1), (var * volumeExtent/seedNumPerAxis/2))
    plt.savefig(outDir+f"1_meanStrain.png")