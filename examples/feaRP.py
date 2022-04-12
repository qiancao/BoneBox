"""

Finite element analysis for rod-plate models

Qian Cao

based on examples/test_20200224_Rxyz

"""

# The usual suspects
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
                             barAreas, shellThicknesses, shellFiberAngles, computeFaceNeighbors = False,
                             verticesScaling=None, plateThicknessScaling=None, barAreaScaling=None):
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
    
    # If needed, scale model for numerical stability.
    if verticesScaling is None:
        verticesScaling = 1.
        
    if plateThicknessScaling is None:
        plateThicknessScaling = 1.
        
    if barAreaScaling is None:
        barAreaScaling = 1.
    
    vertices = vertices * verticesScaling
    shellThicknesses = shellThicknesses * plateThicknessScaling
    barAreas = barAreas * barAreaScaling
    
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
        
    # Scale vertices back to original value
    vertices1 = vertices1 / verticesScaling
    
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


