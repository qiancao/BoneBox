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
    plotter.add_mesh(mesh,scalars=faceOpacity, opacity=1, show_scalar_bar=True, show_edges=True)

    plotter.link_views()
    
    # plotter.show()
    
    plotter.camera_position = [(0.027319263101599902, -0.026339552043427594, 0.013127793077424376),
     (-4.738350340152816e-05, -1.0006934576238784e-05, 6.728825879808234e-05),
     (-0.23312363115051704, 0.22676760723875422, 0.9456372586284911)]
    
    # print(f"saving to f{imgName}...")
    plotter.show(screenshot=f'{imgName}')
    
    # print(plotter.camera_position)
    
    def sdf_show(sdf):
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(cp.asnumpy(sdf.reshape(dims)[sdf.shape[0]//2,:,:]).T)
        ax[1].imshow(cp.asnumpy(sdf.reshape(dims)[:,sdf.shape[1]//2,:]).T)
        ax[2].imshow(cp.asnumpy(sdf.reshape(dims)[:,:,sdf.shape[2]//2]))
        return fig, ax
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import pyvista as pv
    import vtk
    import os
    
    plt.close("all")
    
    # Output directory
    outDir = "/data/BoneBox-out/test_20220308_model_update/"
    os.makedirs(outDir, exist_ok=True)
    
    # Volume initialization parameters
    volumeExtent = 0.01 # meters = 1 cm
    seedNumPerAxis = 6 # Number
    perturbationRadiusScale = 0.8
    randStates = np.arange(5)
    
    # TODO: These scaling factors will have to be revisited
    plateThicknessScaling = 1 #1e-10 #1e-21
    verticesScaling = 1e6
    
    # Generate Phantoms    
    seedPerturbRadius = volumeExtent/seedNumPerAxis/2 * perturbationRadiusScale # meters = 600 um
    
    iters = 10
    displacements = np.zeros((iters+1,len(randStates)))
    stiffnesses = np.zeros((iters+1,len(randStates)))
    
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
        stiffness = np.linalg.norm(forceVector) / displacement
        
        # Random State
        np.save(outDir+f"displacement_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}", displacement)
        np.save(outDir+f"strainsShell4Face_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}", strainsShell4Face)
        
        displacements[0,rr] = displacement
        stiffnesses[0,rr] = stiffness
        
        #%% Iterate
        
        def update_thickness(facesFE, shellThicknesses, strainsShellQuad, threshold_remove, threshold_decrease, threshold_increase, dt):
            
            shellThicknesses[strainsShellQuad < threshold_decrease] = np.maximum(shellThicknesses[strainsShellQuad < threshold_decrease] - dt,0)
            shellThicknesses[strainsShellQuad >= threshold_increase] += dt
            
            ind = strainsShellQuad >= threshold_remove # keep only faces above the threshold_remove
            facesFE = facesFE[ind,:]
            shellThicknesses = shellThicknesses[ind]
            strainsShellQuad = strainsShellQuad[ind]
            
            return facesFE, shellThicknesses, strainsShellQuad
        
        threshold_increase = 1.2e-6
        threshold_decrease = 2.3e-8
        threshold_remove = 2e-13
        dt = 15e-6
        
        for it in range(iters):
            
            verticesFE1, strainsBar, strainsShell = computeFEARodPlateLinear(verticesFE, edges, facesFE,
                                         verticesForce, verticesFixed, forceVector,
                                         elasticModulus, poissonRatio, density,
                                         barAreas, shellThicknesses, shellFiberAngles,
                                         verticesScaling=verticesScaling,
                                         plateThicknessScaling=plateThicknessScaling)
            
            # Mean Strain for all faces
            quadrature = lambda q: np.sqrt(q[:,0]**2 + q[:,1]**2+q[:,2]**2)
            strainsShellQuad = quadrature(strainsShell)
            
            # renderFEA(outDir+f"fea_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}_iter_{it}.png",
            #           strainsBar, strainsShell4Face, v, verticesForce, verticesFixed)
            np.save(outDir+f"displacement_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}_iter_{it}", displacement)
            np.save(outDir+f"strainsShell4Face_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}_iter_{it}", strainsShell4Face)
            
            displacements[it+1,rr] = np.mean(linalg.norm(verticesFE1[verticesForce,:] - verticesFE[verticesForce,:],axis=1))
            stiffnesses[it+1,rr] = np.linalg.norm(forceVector) / displacements[it+1,rr]
            
            plt.figure()
            plt.plot(strainsShellQuad,'k.');
            plt.yscale('log');
            plt.xlabel("Plate Index")
            plt.ylabel("Shell Strain")
            plt.ylim([1e-16, 1e2])
            plt.savefig(outDir+f"strains_{perturbationRadiusScale}_{seedNumPerAxis}_{rr}_iter_{it}.png")
            plt.close("all")
            
            # assert it<0
            
            facesFE, shellThicknesses, previousStrains = update_thickness(facesFE, shellThicknesses, strainsShellQuad, threshold_remove, threshold_decrease, threshold_increase, dt)
            
            #%%
            
            # # Convert to volume
            # dims = (200,200,200)
            # spacing = np.array(Sxyz) / np.array(dims)
            
            # sdf = sdf_rod_plate(verticesFE, facesFE[:,:3], shellThicknesses, edges, [], dims, spacing, k=2)
            # sdf_platten = addPlatten(sdf, 30, plattenValue=0, airValue=np.max(sdf), trimVoxels=10)
            # sdf_platten = set_sdf_boundary(sdf_platten)
            # vertices_sdf, faces_sdf, normals, values = Voxel2SurfMesh(sdf_platten, voxelSize=spacing, origin=None, level=0, step_size=1, allow_degenerate=False)
            # tmesh = trimesh.Trimesh(vertices_sdf, faces=faces_sdf)
            # surf = pv.wrap(tmesh)
            # surf.plot()
            
            #%%
            
            # nodes, elements, tet = Surf2TetMesh(vertices_sdf, faces_sdf, order=1, verbose=1)
            
            # feaResult = computeFEACompressLinear(nodes, elements, Lsim/2, \
            #                                  elasticModulus=17e9, poissonRatio=0.3, \
            #                                  force_total = 1, solver="ParadisoMKL")
                
            # elasticModulus = computeFEAElasticModulus(feaResult)
            
            # fig, ax = sdf_show(sdf)

            
            # print(np.max(thicknessMultiplier))
            # print(np.min(thicknessMultiplier))

        #%%
        
    plt.figure()
    plt.boxplot(stiffnesses.T)
    plt.xlabel("Iteration")
    plt.ylabel("Rod-Plate Model Stiffness")
