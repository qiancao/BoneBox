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
    barArea = 0.1
    
    # Nodes and Elements
    vertices = np.array([[0,0,0],[0,0,1]]).astype(np.double)
    edges = np.array([[0,1]])
    
    # Boundary Conditions
    forceVector = np.array([0,0,1e8]).astype(np.double) # must be a (3,)
    verticesForce = np.array([1],dtype=np.uint64) 
    verticesFixed = np.array([0],dtype=np.uint64)
    
    system = chrono.ChSystemNSC()
    mesh = fea.ChMesh()
    mesh.SetAutomaticGravity(False)
    
    # Create list of nodes and set to mesh
    nodesList = []
    for ind in range(vertices.shape[0]):
        node = fea.ChNodeFEAxyz(chrono.ChVectorD(vertices[ind,0], \
                                                 vertices[ind,1], \
                                                 vertices[ind,2]))
        nodesList.append(node)
        mesh.AddNode(node) # use 0-based indexing here
    
    # Create list of bar elements and set to mesh
    elementsBarList = []
    for ind in range(edges.shape[0]):
        elementBar = fea.ChElementBar()
        elementBar.SetNodes(nodesList[edges[ind,0]], nodesList[edges[ind,1]])
        elementBar.SetBarDensity(density)
        elementBar.SetBarYoungModulus(elasticModulus)
        elementBar.SetBarArea(barArea)
        mesh.AddElement(elementBar)
    
    # Boundary Condition: External force
    for vertInd in verticesForce:
        nodesList[vertInd].SetForce(chrono.ChVectorD(*forceVector))
    
    # Boundary Condition: Truss with nodes of verticesFixed
    trussFixed = chrono.ChBody()
    trussFixed.SetBodyFixed(True)
    
    # Boundary Condition: Link to fixed truss
    constraint = fea.ChLinkPointFrame()
    constraint.Initialize(nodesList[0], trussFixed)
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