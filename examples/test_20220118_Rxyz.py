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

import sys
sys.path.append("../bonebox/phantoms/") # from examples folder

from TrabeculaeVoronoi import *

NULL_INDEX = np.uint64(-1) # Replaced with None in FEA

def computeTriangularMeshNeighborsArray(faces):
    """
    Find neighboring vertex indices for each face of a triangular mesh
    
    For a boundary element without neighboring elements, the vertex index is NULL_INDEX
    
    Array input assumes all neighboring elements will be found, this is quite slow
    
    """
    
    N = faces.shape[0]
    neighbors = np.empty((N,3), dtype=np.uint64)
    neighbors[:] = NULL_INDEX
    
    # Compute neighbors for each face
    facesSets = [set(x) for x in faces]
    for ind in range(N):
        print(f"finding neighbors {ind} of {N}")
        diffList = [facesSets[x] - facesSets[ind] for x in range(faces.shape[0])]
        neighboringFaces = np.nonzero(np.array([len(x) for x in diffList]) == 1)[0] # one element difference
        for nind in neighboringFaces:
            opposingVertex = faces[ind,:] == (facesSets[ind] - facesSets[nind]).pop()
            neighbors[ind,opposingVertex] = diffList[nind].pop()

    return neighbors

def computeTriangularMeshNeighborsList(vertices, faces):
    """
    TODO: This is not complete, might merge with subdivideFacesLocal
    
    Parses neighboring elements for non-triangular faces
    
    For a boundary element without neighboring elements, the vertex index is NULL_INDEX
    
    Only vertices within the non-triangular shell will be "neighbors".
    
    Some overlap with: subdivideFaces
    
    """
    
    vertices1 = list(copy.deepcopy(vertices))
    faces1 = set() # this is a set of (6,) arrays, with neighbors assigned
    faces0 = set([tuple(x) for x in faces])
    
    for ind, face in enumerate(faces0):
        if len(face) == 3:
            faces1.add((*face,0,0,0)) # the face is already triangular
        elif len(face) > 3:
            centroid = np.mean(vertices[face,:],axis=0)
            vertices1.append(centroid) # this is vertices[-1]
            indC = len(vertices1)-1
            for vrt in range(len(face)-1):
                faces1.add((face[vrt], face[vrt+1], indC, ))
            faces1.add((face[vrt+1], face[0], indC))
    
    vertices1 = np.array(vertices1)
    faces1 = np.array(list(faces1))
    
    return vertices1, faces1
    
def computeFEARodPlateLinear(vertices, edges, faces,
                             verticesForce, verticesFixed, forceVector,
                             elasticModulus, poissonRatio, density,
                             barAreas, shellThicknesses, shellFiberAngles, computeFaceNeighbors = False):
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
        
    Added on 20220106: Support for large (locally non-triangular) faces.
        
    """
    
    # Indices should always be np.uint64
    asUint64 = lambda x: x.astype(np.uint64)
    asDouble = lambda x: x.astype(np.double)
    vec2arr = lambda vec: np.array([vec.x, vec.y, vec.z])
    
    # Convert boolean arrays to node indices
    vertices, edges = asDouble(vertices), asUint64(edges)
    if verticesForce.dtype == bool:
        verticesForce = np.nonzero(verticesForce)[0]
    if verticesFixed.dtype == bool:
        verticesFixed = np.nonzero(verticesFixed)[0]
    
    # Find neighboring vertices for elements in faces
    if len(faces) > 0:
        if type(faces) == np.ndarray and faces.shape[1] == 3: # neighbors not defined
            faces = asUint64(faces)
            facesNeighbors = computeTriangularMeshNeighborsArray(faces)
        if type(faces) == np.ndarray and faces.shape[1] == 6: # neighbors already assigned
            facesNeighbors = faces[:,3:]
            faces = faces[:,:3]
        elif type(faces) == list and computeFaceNeighbors == True: # faces is a list and no neighbors are computed
            vertices, faces, facesNeighbors = computeTriangularMeshNeighborsList(faces)
    
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
                              shellFiberAngles[ind] * chrono.CH_C_DEG_TO_RAD, # fiber angle (not used)
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
    for ind in verticesForce:
        constraint = fea.ChLinkPointFrame()
        constraint.Initialize(nodesList[ind], trussForce)
        constraintsForceList.append(constraint)
        system.Add(constraint)
        
    # Boundary Condition: Link trussForce and trussFixed to Prismatic Joint (displacement Z only)
    # TODO: Reimplement this direct ChCoordsysD in the direction of loading
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
    
    # Save Shell Element Strains
    strainsShell = np.zeros((faces.shape[0],3), dtype=np.double) # [m (bending), n(stretching)]
    for ind, element in enumerate(elementsShellList):
        Fi = chrono.ChVectorDynamicD(element.GetNdofs())
        element.ComputeInternalForces(Fi)
        strainsShell[ind,:] = vec2arr(element.e)
        # strainsShell[ind,:] = [element.m.x, element.m.y, element.m.z,
        #                        element.n.x, element.n.y, element.n.z]
        
    # Save Bar Element Strains
    strainsBar = np.zeros(edges.shape[0],dtype=np.double)
    for ind, element in enumerate(elementsBarList):
        strainsBar[ind] = element.GetStrain()
        
    # returns vertex location, strain in Bar, strain in Shell
    return vertices1, strainsBar, strainsShell

def padPolyData(vertexLists):
    """
    Pad faces or edges with number of points for PolyData plotting using pyvista
    """
    return np.hstack([[len(y)]+list(y) for y in vertexLists])

def filterPointsLimXYZ(vertices, xlim, ylim, zlim):
    """
    Filter out vertices outside predefined lims.
    returns index of vertices that are within xyz-lims.
    """
    inrange = lambda x, lim: np.logical_and(x >= lim[0], x < lim[1])
    indx, indy, indz = inrange(vertices[:,0], xlim), inrange(vertices[:,1], ylim), inrange(vertices[:,2], zlim)
    return np.nonzero(indx & indy & indz)[0]

def filterByInd(vertexLists, vertexIndices):
    """
    vertexLists is a list of lists of vertice corresponding to either faces or edges.
    vertexIndices is a list of valid vertices generated from filterPointsLimXYZ.
    returns index of edges or faces where all vertices are in vertexIndices.
    """
    valid = [all([x in vertexIndices for x in y]) for y in vertexLists]
    return np.nonzero(valid)[0]

def pruneMesh(vertices, edges, faces):
    """
    Removes vertices not referenced in edges and faces, and adjust vertex indices in edges and faces accordingly
    Note: 
        1. edges and faces should be unique (findUniqueEdgesAndFaces)
        2. vertices and faces have already been filtered (filterPointsLimXYZ and filterByInd)
    returns new set of (vertices, edges, faces) as (np.array, np.array, *lists*)
    """
    used = set()
    [used.update(x) for x in edges]
    [used.update(x) for x in faces]
    used = list(used)
    used.sort()
    
    convert = dict(zip(used,range(len(used)))) # old indices -> new indices
    
    edges1 = [[convert[x] for x in y] for y in edges] # convertable, since it's shaped Nx2
    faces1 = [[convert[x] for x in y] for y in faces]
    vertices1 = vertices[used,:]
    
    return vertices1, edges1, faces1

def subdivideFaces(vertices, faces):
    """
    Break non-triangular faces down into triangular faces using the centroid as an added vertex
    returns new set of vertices (with centriods of nontriangular faces appended at the end)
    both vertices1 and faces1 are now np.arrays
    """
    
    vertices1 = list(copy.deepcopy(vertices))
    faces1 = set()
    faces0 = set([tuple(x) for x in faces]) # pop
    
    for ind, face in enumerate(faces0):
        if len(face) == 3:
            faces1.add(face) # the face isalready triangular
        elif len(face) > 3:
            centroid = np.mean(vertices[face,:],axis=0)
            vertices1.append(centroid) # this is vertices[-1]
            indC = len(vertices1)-1
            for vrt in range(len(face)-1):
                faces1.add((face[vrt], face[vrt+1], indC))
            faces1.add((face[vrt+1], face[0], indC))
    
    vertices1 = np.array(vertices1)
    faces1 = np.array(list(faces1))
    
    return vertices1, faces1

def reorderFaces(vertices,face):
    """
    Reorders vertex indices in face according to polar angle
    
    https://stackoverflow.com/questions/47949485/sorting-a-list-of-3d-points-in-clockwise-order
    https://stackoverflow.com/questions/44416555/efficient-3d-signed-angle-calculation-in-python-using-arctan2
    https://stackoverflow.com/questions/10133957/signed-angle-between-two-vectors-without-a-reference-plane
    
    """
    
    def theta(a,b): # angle between two 3D vectors
        s = linalg.norm(np.cross(a,b))
        c = np.dot(a,b)
        return np.arctan2(s,c)
    
    face1 = np.array(copy.deepcopy(face))
    centroid = np.mean(vertices[face1,:],axis=0)
    centroid2vertex = vertices[face1,:] - centroid[None,:]
    angles = [] # the first angle is 0th degree from itself
    for ind, vertInd in enumerate(face):
        angles.append((theta(centroid2vertex[ind,:],centroid2vertex[0,:]),vertInd))
    angles.sort()
    
    return [x[1] for x in angles]
    

def subdivideFacesWithLocalNeighbors(vertices, faces):
    """
    same as subdivideFaces but faces are Nx6 numpy.arrays with neighbors already defined
    
    new vertices corresponding to non-triangular face centers is appended to vertices
    
    non-triangular faces are broken into triangular components, along with neighbors, and assigned to the same faceGroup
    
    returns new set of vertices, faces, and faceGroups []
    
    """
    
    vertices1 = list(copy.deepcopy(vertices))
    faces1 = list()
    faces0 = list([tuple(np.uint64(x)) for x in faces]) # pop
    faceGroups = [] # TODO: This could be preallocated.
    
    take = lambda face, vrt: np.take(face, vrt, mode="wrap")
    
    ind, faces = 0, faces0[0]
    
    for ind, face in enumerate(faces0):
        
        if len(face) == 3:
            
            faces1.append((*face, NULL_INDEX, NULL_INDEX, NULL_INDEX)) # the face is already triangular, no neighbors
            faceGroups.append(ind)
            
        elif len(face) > 3:
            
            face = reorderFaces(vertices,face)
            
            centroid = np.mean(vertices[face,:],axis=0) # centroid of vertices in faceGroup
            vertices1.append(centroid) # this is vertices[indC]
            indC = len(vertices1)-1
            
            for vrt in range(len(face)):
                
                faces1.append((take(face,vrt),   take(face,vrt+1), indC,
                            take(face,vrt+2), take(face,vrt-1), NULL_INDEX))
                faceGroups.append(ind)
    
    vertices1 = np.array(vertices1)
    faces1 = np.array(faces1,dtype=np.uint64)
    
    return vertices1, faces1, faceGroups

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
    plotter.background_color = 'k'
    plotter.enable_anti_aliasing()
    
    plotter.subplot(0, 0)
    plotter.add_text("Voronoi Skeleton (Rods and Plates)", font_size=24)
    plotter.add_mesh(rods, show_edges=True)
    plotter.add_mesh(movingVertices,color='y',point_size=10)
    plotter.add_mesh(fixedVertices,color='c',point_size=10)
    
    plotter.subplot(0, 1)
    plotter.add_text("Strain Plates Only", font_size=24)
    plotter.add_mesh(movingVertices,color='y',point_size=10)
    plotter.add_mesh(fixedVertices,color='c',point_size=10)
    
    mesh = pv.PolyData(v.vertices, padPolyData(v.faces))
    plotter.add_mesh(mesh,scalars=faceOpacity,opacity=faceOpacity, show_scalar_bar=True)

    plotter.link_views()

    plotter.camera_position = [(-15983.347882469203, -25410.916652156728, 9216.573794734646),
     (0.0, 0.0, 0.0),
     (0.16876817270434966, 0.24053571467115548, 0.9558555716475535)]
    
    # print(f"saving to f{imgName}...")
    plotter.show(screenshot=f'{imgName}')
    
    # print(plotter.camera_position)
   
# def plates2image(vertices, faces, thicknesses , dims, spacing):
#     """
    
#     thicknesses: radii of plates in voxels, same len as faces
#     dims: number of voxels in xyz
#     spacing: voxel size
    
#     """
    
#     image = np.zeros(dims, dtype=np.uint8)
    
#     for ind, face in enumerate(faces):
        
#         full_surface(verts)
        
#         # print(f"{ind}/{len(faces)}")
        
#         # grid = pv.UniformGrid(
#         #     dims = np.array(dims),
#         #     spacing = np.array(spacing),
#         #     origin = -(np.array(dims)*np.array(spacing))/2,
#         # )
        
#         # mesh = pv.PolyData(vertices, padPolyData([face]))
        
#         # grid_dist = grid.compute_implicit_distance(mesh)
        
#         # dist = grid_dist.point_data['implicit_distance']
#         # dist = np.array(dist).reshape(dims)
        
#         # dist = (np.abs(dist) < thicknesses[ind]).astype(np.uint8)
    
#         # image = np.maximum(image, dist)
        
#     return image
   
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import pyvista as pv
    import vtk
    import os
    
    # Output directory
    outDir = "/data/BoneBox-out/test_20220118_Rxyz/"
    os.makedirs(outDir, exist_ok=True)
    
    #%%
    
    # TODO: These scaling factors will have to be revisited
    thicknessScaling = 1e-21
    verticesScaling = 1e6
    
    # interesting, monotonic relationship only observed smaller than 1e-8
    # var = np.linspace(50e-6, 500e-6, 5) # So, this needs to be scaled by 1e6
    var = np.linspace(0.2, 1, 5)
    varName = "Perturbation Radius (um)"
    
    # random states
    rands = np.arange(3)
    
    # save displacement and strains
    displacements = []
    strains = []
    
    # Volume initialization parameters
    volumeExtent = 0.01 # meters = 1 cm
    seedNumPerAxis = 8 # Number
    
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
            thicknesses0, radii0 = 100e-6*thicknessScaling, 100e-6 # Range of Thickness IS the determining factor of monotonic relationship
            elasticModulus = 17e9 # Pa
            poissonRatio = 0.3
            density = 0 # NO_EFFECT: Density has no impact on results
            barAreas = np.pi*radii0**2*np.ones(len(edges),dtype=float) # m**2 = PI*r**2
            shellThicknesses = thicknesses0*np.ones(len(facesFE),dtype=float) # 
            shellFiberAngles = np.ones(len(facesFE),dtype=float) # NO_EFFECT: Fiber angle has no impact on results
            
            verticesFE1, strainsBar, strainsShell = computeFEARodPlateLinear(verticesFE, edges, facesFE,
                                         verticesForce, verticesFixed, forceVector,
                                         elasticModulus, poissonRatio, density,
                                         barAreas, shellThicknesses, shellFiberAngles)
            
            # Mean Strain for all faces
            quadrature = lambda q: np.sqrt(q[:,0]**2 + q[:,1]**2+q[:,2]**2)
            strainsShellQuad = quadrature(strainsShell)
            
            strainsShellQuadCombined = []
            faceGroupArr = np.array(faceGroup)
            for g in range(np.max(faceGroup)+1):
                strainsShellQuadCombined.append(np.mean(strainsShellQuad[faceGroupArr == g]))
            strainsShell4Face = np.array(strainsShellQuadCombined)
            
            # Render volume
            renderFEA(outDir+f"fea_{vv}_{rr}.png", strainsBar, strainsShell4Face, v, verticesForce, verticesFixed)
        
            displacement = np.mean(linalg.norm(verticesFE1[verticesForce,:] - verticesFE[verticesForce,:],axis=1))
            
            # Random State
            displacementRands.append(displacement)
            strainsRands.append(np.median(strainsShell4Face))
            
            # Convert to volume
            dims = (250,250,250)
            spacing = np.array(Sxyz) / np.array(dims)
            
            thicknessVoxels = (shellThicknesses/thicknessScaling) / spacing[0]
            print(thicknessVoxels)
            
            # image = plates2image(vertices, faces, (thicknessVoxels,)*len(faces), dims, spacing)
            
            
            assert False
            
            # assert False
        
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

    # #%% Face statistics
    
    # # theta = lambda v1,v2: np.arccos(np.dot(v1,v2)) / (linalg.norm(v1)*linalg.norm(v2))
    # theta = lambda v1,v2: np.arccos(np.dot(v1/linalg.norm(v1), v2/linalg.norm(v2)))
    
    # faceThetas = []
    # faceAreas = []
    # for ind, face in enumerate(faces):
    #     faceVertices = vertices[face,:]
    #     normal = computeFaceNormals([faceVertices])
    #     faceThetas.append(theta(normal,forceVector)[0])
    #     faceAreas.append(computeFaceAreas([faceVertices]))
        
    # faceThetas = np.abs(np.array(faceThetas).T - np.pi/2)
        
    # plt.figure()
    # plt.plot(faceThetas,strainsShell4Face,'bv')
    # plt.xlim([0,np.pi/2])
    # plt.xlabel("Angle with Principal Loading Direction")
    # plt.ylim([0,np.median(strainsShell4Face)*2])
    # plt.ylabel("Quad Sum of Principal Strains")
    
    # plt.figure()
    # plt.plot(faceAreas,strainsShell4Face,'bv')
    # plt.xlim([0,200])
    # plt.xlabel("Plate Area")
    # plt.ylim([0,np.median(strainsShell4Face)*2])
    # plt.ylabel("Quad Sum of Principal Strains")