"""

Notes: 20220105

use np.max(np.abs(element.e)) for strain

elementShell.AddLayer(shellThicknesses[0],
                      90 * chrono.CH_C_DEG_TO_RAD, # fiber angle (not used)
                      materialKirchoffShell)

fiber angle does not seem to affect strain

"""

import numpy as np
import pychrono as chrono
import pychrono.fea as fea
import pychrono.pardisomkl as mkl
import matplotlib.pyplot as plt

def test_ChElementShellBST(thickness):

    # Nodes and Elements
    vertices = np.array([[0,0,0],[1,0,0],[0.5,0,1]]).astype(np.double)
    faces = np.array([[0,1,2]])
    
    # Boundary Conditions
    forceVector = np.array([0,0,+1e3]).astype(np.double) # must be a (3,)
    verticesForce = np.array([2],dtype=np.uint64)
    verticesFixed = np.array([0,1],dtype=np.uint64)
    
    # Material Properties
    elasticModulus = 1e9 # N/m**2
    poissonRatio = 0.3
    density = 1.6e3 # [1.6-2 g/cm**3] not used
    shellThicknesses = thickness*np.ones(faces.shape[0],dtype=float) # m
    
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
        
    # Add shell element
    elementShell = fea.ChElementShellBST()
    elementShell.SetNodes(nodesList[faces[0,0]],
                     nodesList[faces[0,1]],
                     nodesList[faces[0,2]],
                     None, None, None)
    elementShell.AddLayer(shellThicknesses[0],
                          90 * chrono.CH_C_DEG_TO_RAD, # fiber angle (not used)
                          materialKirchoffShell)
    mesh.AddElement(elementShell)
    
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
    for ind in verticesForce:
        constraint = fea.ChLinkPointFrame()
        constraint.Initialize(nodesList[ind], trussForce)
        constraintsForceList.append(constraint)
        system.Add(constraint)
        
    # Boundary Condition: Link trussForce and trussFixed to Prismatic Joint (displacement Z only)
    constraint = chrono.ChLinkLockPrismatic()
    constraint.Initialize(trussFixed, trussForce, chrono.ChCoordsysD())
    system.AddLink(constraint)
    
    # Boundary Condition: Link to fixed truss
    constraintsFixedList = []
    for ind in verticesFixed:
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
    
    Fi = chrono.ChVectorDynamicD(elementShell.GetNdofs())
    elementShell.ComputeInternalForces(Fi)
    strains = [elementShell.m.x, elementShell.m.y, elementShell.m.z, # bending?
               elementShell.n.x, elementShell.n.y, elementShell.n.z] # stretching?

    return vertices, vertices1, strains, elementShell
   
if __name__ == "__main__":
    
    plt.close('all')
    
    thicknesses = np.linspace(10e-6,1000e-6,200) # This is now thickness of shell elements
    shellStrains = np.zeros((thicknesses.shape[0],6))
    vertexPositions1 = []
    elements = []
   
    for ind, thickness in enumerate(thicknesses):
        
        vertices0, vertices1, strainsShell, element = test_ChElementShellBST(thickness)
        
        shellStrains[ind,:] = strainsShell
        vertexPositions1.append(vertices1) # FEA-computed position
        elements.append(element)
    
    # Shell strains vs shell thickness
    vec2arr = lambda vec: np.array([vec.x, vec.y, vec.z])
    legend = ["mx",'my','mz','nx','ny','nz']
    fig = plt.figure()
    # for ind in range(6):
        # plt.plot(thicknesses, shellStrains[:,ind])
    plt.plot(thicknesses,[np.max(np.abs(vec2arr(elements[x].e))) for x in range(len(thicknesses))])
    plt.legend(legend)
    plt.xlabel("Shell Thickness")
    plt.ylabel("Plate Strain")
    
    # Shell strains vs shell thickness
    arr2xyz = lambda arr: (arr[:,0],arr[:,1],arr[:,2])
    colors = plt.cm.viridis(np.linspace(0,1,len(thicknesses)))    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(*arr2xyz(vertices0), 'ko')
    for ind, thickness in enumerate(thicknesses):
        ax.plot3D(*arr2xyz(vertexPositions1[ind]), 'v',color = colors[ind,:])
