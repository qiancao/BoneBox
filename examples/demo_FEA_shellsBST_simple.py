#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 05:27:12 2021

@author: qcao
"""

import numpy as np
import pychrono as chrono
import pychrono.fea as fea
import pychrono.pardisomkl as mkl

if __name__ == "__main__":
    
    # Material Properties
    elasticModulus = 17e9
    poissonRatio = 0.3
    density = 0.1
    
    # Nodes and Elements
    vertices = np.array([[0,0,0],[1,0,0],[0,0,1]]).astype(np.double)
    faces = np.array([[0,1,2]])
    
    # Boundary Conditions
    forceVector = np.array([0,0,1e6]).astype(np.double) # must be a (3,)
    verticesForce = np.array([2],dtype=np.uint64) 
    verticesFixed = np.array([0,1],dtype=np.uint64)
    
    facesThickness = 0.001
    
    system = chrono.ChSystemNSC()
    mesh = fea.ChMesh()
    mesh.SetAutomaticGravity(False)
    
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
    
    # Elements
    elementShell = fea.ChElementShellBST()
    elementShell.SetNodes(nodesList[faces[0,0]],
                     nodesList[faces[0,1]],
                     nodesList[faces[0,2]],
                     None,
                     None,
                     None)
    elementShell.AddLayer(facesThickness,
                          0 * chrono.CH_C_DEG_TO_RAD, # fiber angle (not used)
                          materialKirchoffShell)
    mesh.AddElement(elementShell)
    
    # Set Force
    nodesList[2].SetForce(chrono.ChVectorD(*forceVector))
    
    # Boundary Condition: Truss with nodes of verticesFixed
    trussFixed = chrono.ChBody()
    trussFixed.SetBodyFixed(True)
    
    # Boundary Condition: Link to fixed truss
    constraintsFixedList = []
    for ind in range(len(verticesFixed)):
        constraint = fea.ChLinkPointFrame()
        constraint.Initialize(nodesList[ind], trussFixed)
        constraintsFixedList.append(constraint)
        system.Add(constraint)
        
    # Prepare system and solve
    system.Add(mesh)
    system.Add(trussFixed)
    
    msolver = mkl.ChSolverPardisoMKL()
    msolver.LockSparsityPattern(True)
    
    # TODO node positions before solve
    for ind in range(len(nodesList)):
        pos = nodesList[ind].GetPos()
        print(pos)
    print()
        
    # Solve
    system.SetSolver(msolver)
    system.DoStaticLinear()
    
    # TODO node positions after solve
    for ind in range(len(nodesList)):
        pos = nodesList[ind].GetPos()
        print(pos)