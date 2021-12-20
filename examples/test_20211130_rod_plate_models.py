#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 17:11:02 2021

@author: qcao

sketch of rod plate model and skeleton code for PyChrono

Useful links:
    https://github.com/projectchrono/chrono/blob/03931da6b00cf276fc1c4882f43b2fd8b3c6fa7e/src/chrono/fea/ChElementShellBST.cpp
    

"""

# the usual suspects
import numpy as np
import nrrd

# meshing
from skimage import measure
import tetgen
import trimesh
import pyvista as pv

# finite element library
import pychrono as chrono
import pychrono.fea as fea
import pychrono.pardisomkl as mkl

def computeTriangularMeshNeighbors(faces):
    """
    Find neighboring vertex indices for each face of a triangular mesh
    
    For a boundary element without neighboring elements, the vertex index is NULL_INDEX
    """
    
    NULL_INDEX = np.uint64(-1)
    N = faces.shape[0]
    neighbors = np.empty((N,3), dtype=np.uint64)
    neighbors[:] = NULL_INDEX
    
    # Compute neighbors for each face
    facesSets = [set(x) for x in faces]
    for ind in range(N): 
        diffList = [facesSets[x] - facesSets[ind] for x in range(faces.shape[0])]
        neighboringFaces = np.nonzero(np.array([len(x) for x in diffList]) == 1)[0] # one element difference
        for nind in neighboringFaces:
            opposingVertex = faces[ind,:] == (facesSets[ind] - facesSets[nind]).pop()
            neighbors[ind,opposingVertex] = diffList[nind].pop()

    return neighbors, NULL_INDEX

def computeFEARodPlateLinear(vertices, edges, faces,
                             verticesForce, verticesFixed, forceVector,
                             elasticModulus, poissonRatio, density,
                             barAreas, shellThicknesses):
    """
    Finite element analysis for mixed rod-plate model
    
    elasticModulus, poissonRatio
    
    vertices : (numVerts,3) --> ChNodeFEAxyz
    edges : (numEdges,2) --> ChElementBar
    faces : (numFaces,3), or (numFaces,6) --> ChElementShellBST 0-1-2 (3-4-5 neighboring triangles opposed to 0-1-2)
    
    forceVector : (3,) force direction and magnitude
    verticesForce : applied force (array of bools or node indices)
    verticesFixed : fixed nodes (array of bools or node indices)
    
    For an example on ChElementShellBST, see:
        https://github.com/projectchrono/chrono/blob/develop/src/demos/python/fea/demo_FEA_shellsBST.py
    
    """
    
    # Indices should always be np.uint64
    asUint64 = lambda x: x.astype(np.uint64)
    asDouble = lambda x: x.astype(np.double)
    
    # Convert boolean arrays to node indices
    vertices, edges, faces = asDouble(vertices), asUint64(edges), asUint64(faces)
    if verticesForce.dtype == bool:
        verticesForce = np.nonzero(verticesForce)[0]
    if verticesFixed.dtype == bool:
        verticesFixed = np.nonzero(verticesFixed)[0]
    
    # Find neighboring vertices for elements in faces
    facesNeighbors, NULL_INDEX = computeTriangularMeshNeighbors(faces)
    
    # System and mesh
    system = chrono.ChSystemNSC()
    
    mesh = fea.ChMesh()
    mesh.SetAutomaticGravity(False)
    
    # Material (Shell and Bars are different)
    materialKirchoff = fea.ChElasticityKirchhoffIsothropic(elasticModulus, poissonRatio)
    materialKirchoffShell = fea.ChMaterialShellKirchhoff(materialKirchoff)
    materialKirchoffShell.SetDensity(density)

    # Create list of nodes and set to mesh
    nodesList = []
    for ind in range(vertices.shape[0]):
        node = fea.ChNodeFEAxyz(chrono.ChVectorD(vertices[ind,0], \
                                                 vertices[ind,1], \
                                                 vertices[ind,2]))
        nodesList.append(node)
        mesh.AddNode(node) # use 0-based indexing here
        
    # Create list of shell elements and set to mesh
    elementsShellList = []
    for ind in range(faces.shape[0]):
        
        # Get neighboring nodes, None if on face boundary
        neighboringNodeInds = list(facesNeighbors[ind,:])
        neighboringNodes = [None if x==NULL_INDEX else nodesList[x] for x in neighboringNodeInds]
        
        # Add shell element
        elementShell = fea.ChElementShellBST()
        elementShell.SetNodes(nodesList[faces[ind,0]],
                         nodesList[faces[ind,1]],
                         nodesList[faces[ind,2]],
                         neighboringNodes[0],
                         neighboringNodes[1],
                         neighboringNodes[2])
        elementShell.AddLayer(shellThicknesses[ind],
                              0 * chrono.CH_C_DEG_TO_RAD, # fiber angle (not used)
                              materialKirchoffShell)
        elementsShellList.append(elementShell)
        mesh.AddElement(elementShell)
    
    # Create list of bar elements and set to mesh
    elementsBarList = []
    for ind in range(edges.shape[0]):
        elementBar = fea.ChElementBar()
        elementBar.SetNodes(nodesList[edges[ind,0]], nodesList[edges[ind,1]])
        elementBar.SetBarDensity(density)
        elementBar.SetBarYoungModulus(elasticModulus)
        elementBar.SetBarArea(barAreas[ind])
        elementsBarList.append(elementBar)
        mesh.AddElement(elementBar)
        
    # Boundary Condition: Truss with nodes of verticesForce
    trussForce = chrono.ChBody()
    
    # Boundary Condition: Truss with nodes of verticesFixed
    trussFixed = chrono.ChBody()
    trussFixed.SetBodyFixed(True)

    # Boundary Condition: External force (****This took a long night to debug)
    for vertInd in verticesForce:
        nodesList[vertInd].SetForce(chrono.ChVectorD(*forceVector))
        
    # Boundary Consition: Link to moving truss (should move in unison)
    constraintsForceList = []
    for ind in range(len(verticesForce)):
        constraint = fea.ChLinkPointFrame()
        constraint.Initialize(nodesList[ind], trussForce)
        constraintsForceList.append(constraint)
        system.Add(constraint)
        
    # Boundary Condition: Link trussForce and trussFixed to Prismatic Joint (displacement Z only)
    # TODO: Reimplement this direct ChCoordsysD in the direction of loading
    # constraint = chrono.ChLinkLockPrismatic()
    # constraint.Initialize(trussFixed, trussForce, chrono.ChCoordsysD())
    # system.AddLink(constraint)
    
    # Boundary Condition: Link to fixed truss
    constraintsFixedList = []
    for ind in range(len(verticesFixed)):
        constraint = fea.ChLinkPointFrame()
        constraint.Initialize(nodesList[ind], trussFixed)
        constraintsFixedList.append(constraint)
        system.Add(constraint)
    
    # Prepare system and solve
    system.Add(mesh)
    system.Add(trussForce)
    system.Add(trussFixed)
    
    # Solver
    msolver = mkl.ChSolverPardisoMKL()
    msolver.LockSparsityPattern(True)
        
    # Solve
    system.SetSolver(msolver)
    system.DoStaticLinear()
    
    # Node positions of solution
    vertices1 = vertices.copy()
    for ind in range(len(nodesList)):
        pos = nodesList[ind].GetPos()
        vertices1[ind,0] = pos.x
        vertices1[ind,1] = pos.y
        vertices1[ind,2] = pos.z
    
    # Save Shell Element Strains
    strainsShell = np.zeros((faces.shape[0],6), dtype=np.double) # [m (bending), n(stretching)]
    for ind, element in enumerate(elementsShellList):
        Fi = chrono.ChVectorDynamicD(element.GetNdofs())
        element.ComputeInternalForces(Fi)
        strainsShell[ind,:] = [element.m.x, element.m.y, element.m.z,
                               element.n.x, element.n.y, element.n.z]
        
    # Save Bar Element Strains
    strainsBar = np.zeros(edges.shape[0],dtype=np.double)
    for ind, element in enumerate(elementsBarList):
        strainsBar[ind] = element.GetStrain()
        
    return vertices1, strainsBar, strainsShell
        
if __name__ == "__main__":
    
    # Nodes and Elements
    vertices = np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,1]]).astype(np.double)
    # edges = np.array([[0,2],[1,3]])
    edges = np.array([[0,2]])
    faces = np.array([[0,1,2],[2,1,3]])
    
    # Boundary Conditions
    forceVector = np.array([0,0,-1.]).astype(np.double) # must be a (3,)
    verticesForce = np.array([2,3],dtype=np.uint64)
    verticesFixed = np.array([0,1],dtype=np.uint64)
    
    # Material Properties
    elasticModulus = 17e9 # Pa
    poissonRatio = 0.3
    density = 0.1
    barAreas = np.pi*100e-6**2*np.ones(edges.shape[0],dtype=float) # m**2 = PI*r**2
    shellThicknesses = 100e-6*np.ones(faces.shape[0],dtype=float) # m
    
    vertices1, strainsBar, strainsShell = computeFEARodPlateLinear(vertices, edges, faces,
                                 verticesForce, verticesFixed, forceVector,
                                 elasticModulus, poissonRatio, density,
                                 barAreas, shellThicknesses)

    from mpl_toolkits import mplot3d
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    

