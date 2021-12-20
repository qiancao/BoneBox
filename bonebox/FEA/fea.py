# -*- coding: utf-8 -*-

"""
FEA.fea

Utilities module for finite element analysis (pychrono backend)

Author:  Qian Cao

"""

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

def addPlatten(volume, plattenThicknessVoxels):
    # adds compression plates in Z
    # leaves a single-voxel space at the edge of volume (for isosurface)
    
    vmax = np.max(volume)
    vmin = np.min(volume)
    
    # Leaves single-voxel space at edge of volume for isosurface ops
    volume[:,:,0] = vmin
    volume[:,:,-1] = vmin
    
    # Define platten
    volume[1:-1,1:-1,1:plattenThicknessVoxels] = vmax
    volume[1:-1,1:-1,-plattenThicknessVoxels:-1] = vmax

    return volume

def Voxel2HexaMeshIndexCoord(volume):
    """
    Directly convert voxels to hexamesh (bricks) and returns mesh in index coordinates
        
    Example: to retrieve nodes corresponding to element 217:
    nodesSortedUnique[elements[217],:]
    
    Given the default voxelSize and origin, coordinates range from (-0.5 to dimXYZ+0.5)
    
    nodesSortedUnique.shape = (nodes,3)
    """
    
    xx, yy, zz = np.nonzero(volume)
    nElements = len(xx)
    
    # Compute list of vertex (node) coordinates
    nodes = np.empty((3,nElements,8)) #(xyz, N, verts)
    nodes[:] = np.NaN
    
    nodes[:,:,0] = np.vstack((xx-0.5, yy-0.5, zz-0.5))
    nodes[:,:,1] = np.vstack((xx+0.5, yy-0.5, zz-0.5))
    nodes[:,:,2] = np.vstack((xx+0.5, yy+0.5, zz-0.5))
    nodes[:,:,3] = np.vstack((xx-0.5, yy+0.5, zz-0.5))
    nodes[:,:,4] = np.vstack((xx-0.5, yy-0.5, zz+0.5))
    nodes[:,:,5] = np.vstack((xx+0.5, yy-0.5, zz+0.5))
    nodes[:,:,6] = np.vstack((xx+0.5, yy+0.5, zz+0.5))
    nodes[:,:,7] = np.vstack((xx-0.5, yy+0.5, zz+0.5))
    
    nodes = np.reshape(nodes, (3,-1))
    
    # Simplify, keep only unique nodes (faster than np.unique)
    nodesSortedInds = np.lexsort(nodes) # last element (z) first
    nodesSorted = nodes[:,nodesSortedInds]
    dnodes = np.sum(np.diff(nodesSorted, axis=1), axis=0) # assumes nodes are sorted lexigraphically
    mask = np.hstack((True, dnodes!=0)) # mask array with unique nodes set to True on first appearance
    nodesSortedUnique = nodesSorted[:,mask] # *** final list of node coordinates
    
    # Initialize array of elements, indices to nodes in nodesSortedUnique
    elements = np.zeros(nElements*8, dtype=np.int64) # final list of elements (index of nodes in nodeSortedUnique)
    nodeIndsOriginal = np.arange(nElements*8)
    nodeIndsOriginalSorted = nodeIndsOriginal[nodesSortedInds]
    nodeIndsReduced = np.cumsum(mask)-1
    elements[nodeIndsOriginalSorted] = nodeIndsReduced
    elements = np.reshape(elements, (nElements, 8))
    
    nodesSortedUnique = nodesSortedUnique.T
    
    return nodesSortedUnique, elements

def HexaMeshIndexCoord2Voxel(nodes, elements, dim):
    """
    Convert hexamesh (bricks) in index coordinates to volume in voxels
    
    dim: dimension of volume in x, y and z in voxels (tuple)
        
    Example: to retrieve nodes corresponding to element 217:
    nodesSortedUnique[elements[217],:]
    
    Given the default voxelSize and origin, coordinates range from (-0.5 to dimXYZ+0.5)
    
    nodesSortedUnique.shape = (nodes,3)
    
    """
    
    volume = np.zeros(dim, dtype=bool) # initialize volume of False
    xyz = nodes[elements,:][:,0,:] + 0.5 # voxel coordinates of bone
    xyz = xyz.astype(int)
    volume[tuple(xyz.T)] = True
    
    return volume

def HexaMeshIndexCoord2VoxelValue(nodes, elements, dim, elementValues):
    """
    Convert hexamesh (bricks) in index coordinates to volume in voxels with value of voxels assigned according to elementValues.
    
    dim: dimension of volume in x, y and z in voxels (tuple)
    elementValues: len(elements) == len(elementValues)
        
    Example: to retrieve nodes corresponding to element 217:
    nodesSortedUnique[elements[217],:]
    
    Given the default voxelSize and origin, coordinates range from (-0.5 to dimXYZ+0.5)
    
    nodesSortedUnique.shape = (nodes,3)
    
    """
    
    volume = np.zeros(dim, dtype=elementValues.dtype) # initialize volume of False
    xyz = nodes[elements,:][:,0,:] + 0.5 # voxel coordinates of bone
    xyz = xyz.astype(int)
    volume[tuple(xyz.T)] = elementValues
    
    return volume

def Index2AbsCoords(nodes, dim, voxelSize=(1,1,1), origin=(0,0,0)):
    """
    Convert array of node coordinates from index coordinates to absolute coordinates
        
        dim: volume dimension in voxels (tuple)
        origin: shift of origin from center of volume
        (0,0,0) corresponds to center of volume (default), (-X/2, -Y/2, -Z/2) refers to "top left corner".
    """

    dim = np.array(dim)
    voxelSize = np.array(voxelSize)
    origin = np.array(origin)
    nodes = (nodes - dim/2) * voxelSize + origin
    return nodes

def Abs2IndexCoords(nodes, dim, voxelSize=(1,1,1), origin=(0,0,0)):
    """
    Convert array of node coordinates from abs coordinates to index coordinates
        
        dim: volume dimension in voxels (tuple)
        origin: shift of origin from center of volume
        (0,0,0) corresponds to center of volume (default), (-X/2, -Y/2, -Z/2) refers to "top left corner".
    """

    dim = np.array(dim)
    voxelSize = np.array(voxelSize)
    origin = np.array(origin)
    nodes = (nodes - origin) / voxelSize + dim/2
    return nodes

def Voxel2SurfMesh(volume, voxelSize=(1,1,1), origin=None, level=None, step_size=1, allow_degenerate=False):
    # Convert voxel image to surface
    
    if level == None:
        level = (np.max(volume))/2
    
    vertices, faces, normals, values = \
        measure.marching_cubes_lewiner(volume = volume, level = level, spacing = voxelSize, \
                                       step_size = step_size, allow_degenerate = allow_degenerate)
    return vertices, faces, normals, values

def Surf2TetMesh(vertices, faces, order=1, verbose=1):
    # Convert surface mesh to tetrahedra
    
    tet = tetgen.TetGen(vertices,faces)
    tet.tetrahedralize(order=order,verbose=verbose)
    
    nodes = tet.node
    elements = tet.elem
    
    return nodes, elements, tet

def isWatertight(vertices, faces):
    # Check if mesh is watertight
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    return mesh.is_watertight

def computeFEACompressLinearHex(nodes, elements, plateThickness, \
                             elasticModulus=17e9, poissonRatio=0.3, \
                             force_total = 1, solver="ParadisoMKL"):
    # TODO: Think about how to refactor this
    # Linear Finite Element Analysis with PyChrono but with hexahedral meshes
    # plateThickness: Thickness of compression plates in absolute units
    
    system = chrono.ChSystemNSC()
    material = chrono.fea.ChContinuumElastic(elasticModulus, poissonRatio)
    mesh = fea.ChMesh()
    
    print("setting nodes")
    node_list = []
    for ind in range(nodes.shape[0]):
        node_list.append(fea.ChNodeFEAxyz(chrono.ChVectorD(nodes[ind,0], nodes[ind,1], nodes[ind,2])))
        mesh.AddNode(node_list[ind]) # use 0-based indexing here
        
    print("setting elements")
    ele_list = []
    for ind in range(elements.shape[0]):
        node0ind, node1ind, node2ind, node3ind, node4ind, node5ind, node6ind, node7ind \
            = elements[ind,0], elements[ind,1], elements[ind,2], elements[ind,3], \
                elements[ind,4], elements[ind,5], elements[ind,6], elements[ind,7]
        ele_list.append(fea.ChElementHexa_8())
        ele_list[ind].SetNodes(node_list[node0ind], node_list[node1ind], node_list[node2ind], node_list[node3ind], \
                               node_list[node4ind], node_list[node5ind], node_list[node6ind], node_list[node7ind])
        ele_list[ind].SetMaterial(material)
        mesh.AddElement(ele_list[ind]) # use 0-based indexing here
        
    mesh.SetAutomaticGravity(False)
    system.Add(mesh)
    
    zmax = np.max(nodes,axis = 0)[2] - plateThickness
    zmin = np.min(nodes,axis = 0)[2] + plateThickness
    
    faceA_nodeind = np.asarray(np.nonzero(nodes[:,2]>zmax))[0]
    faceB_nodeind = np.asarray(np.nonzero(nodes[:,2]<zmin))[0]
    
    # Distribute total force over nodes on face A
    force_dist = force_total / len(faceA_nodeind)
    
    print("setting force")
    for ind in faceA_nodeind:
        node_list[ind].SetForce(chrono.ChVectorD(0,0,-force_dist))
    
    # Face B: Fixed truss
    truss = chrono.ChBody()
    truss.SetBodyFixed(True)
    system.Add(truss)
    
    # Face A: Moving truss
    truss_moving = chrono.ChBody()
    system.Add(truss_moving)
    
    print("setting constraints")
    
    print("... face B: "+str(len(faceB_nodeind)))
    # Add face B nodes to fixed truss
    constr_list = []
    for ind in faceB_nodeind:
        constr_list.append(fea.ChLinkPointFrame())
        constr_list[-1].Initialize(node_list[ind], truss)
        system.Add(constr_list[-1])
    
    print("... face A: "+str(len(faceA_nodeind)))
    # Add face A nodes to moving truss
    constr_list_moving = []
    for ind in faceA_nodeind:
        constr_list_moving.append(fea.ChLinkPointFrame())
        constr_list_moving[-1].Initialize(node_list[ind], truss_moving)
        system.Add(constr_list_moving[-1])
    
    print("... constr_A ChLinkLockPrismatic", flush=True)
    # Prismatic Joint, this works for Paradiso MKL solver
    constr_A = chrono.ChLinkLockPrismatic()
    print("... initializing", flush=True)
    constr_A.Initialize(truss, truss_moving, chrono.ChCoordsysD())
    print("... Adding link", flush=True)
    system.AddLink(constr_A)
    
    print("solving with" + solver, flush=True)
    
    if solver == "ParadisoMKL": # TODO: Validate results from this solver
        msolver = mkl.ChSolverPardisoMKL()
        msolver.LockSparsityPattern(True)
    else: # MINRES solver
        msolver = chrono.ChSolverMINRES()
        msolver.SetMaxIterations(100)
        msolver.SetTolerance(1e-10)
        msolver.EnableDiagonalPreconditioner(True)
        msolver.SetVerbose(True)
    
    system.SetSolver(msolver)
    system.DoStaticLinear()
    
    print("formating output", flush=True)
    ## Format FEA output
    
    # Retrieve nodal position
    node_arr_sol = np.copy(nodes) ###### <<< nodal solution
    for ind in range(nodes.shape[0]):
        pos = node_list[ind].GetPos()
        node_arr_sol[ind,0] = pos.x
        node_arr_sol[ind,1] = pos.y
        node_arr_sol[ind,2] = pos.z
    
    # element centroid positions
    ele_centroid = np.empty([len(ele_list),3]) ##### <<< elemental centroid position
    for ind in range(len(ele_list)):
        sumxyz = np.zeros([1,3]) # [x,y,z] for centroid
        for iind in range(4): # iterate over 4 nodes
            nodeind = elements[ind,iind] # all zero-based indices now
            sumxyz += nodes[nodeind,0:3]
        sumxyz = sumxyz / 4 # center of 4 nodes
        ele_centroid[ind,:] = sumxyz
        
    ele_Tstress = np.empty((len(ele_list),6))
    ele_Tstrain = np.empty((len(ele_list),6))
    ele_VMstress = np.empty((len(ele_list),1))
    
    for ind in range(len(ele_list)):
        
        ep = ele_list[ind].GetStrain(0,0,0)
        sig = ele_list[ind].GetStress(0,0,0)
        sigVM = sig.GetEquivalentVonMises()
    
        strain_vector = np.array([ep.XX(), ep.YY(), ep.ZZ(), ep.XY(), ep.XZ(), ep.YZ()])
        stress_vector = np.array([sig.XX(), sig.YY(), sig.ZZ(), sig.XY(), sig.XZ(), sig.YZ()])
    
        ele_Tstress[ind,:] = stress_vector.reshape((1,6))
        ele_Tstrain[ind,:] = strain_vector.reshape((1,6))
        ele_VMstress[ind,:] = sigVM
        
    feaResult = {
        "nodes" : nodes,
        "displacement" : node_arr_sol - nodes,
        "elementCentroids" : ele_centroid,
        "nodeIndA" : faceA_nodeind,
        "nodeIndB" : faceB_nodeind,
        "elementVMstresses" : ele_VMstress,
        "elementStresses" : ele_Tstress,
        "elementStrains" : ele_Tstrain,
        "force" : force_total
        }
    
    del mesh, solver, system
    
    return feaResult
    
def computeFEACompressLinear(nodes, elements, plateThickness, \
                             elasticModulus=17e9, poissonRatio=0.3, \
                             force_total = 1, solver="ParadisoMKL"):
    # TODO: Think about how to refactor this
    # Linear Finite Element Analysis with PyChrono
    # plateThickness: Thickness of compression plates in absolute units

    system = chrono.ChSystemNSC()
    material = chrono.fea.ChContinuumElastic(elasticModulus, poissonRatio)
    mesh = fea.ChMesh()

    # TODO: is it necessary to add nodes to mesh?
    node_list = []
    for ind in range(nodes.shape[0]):
        node_list.append(fea.ChNodeFEAxyz(chrono.ChVectorD(nodes[ind,0], nodes[ind,1], nodes[ind,2])))
        mesh.AddNode(node_list[ind]) # use 0-based indexing here
    
    ele_list = []
    for ind in range(elements.shape[0]):
        node0ind, node1ind, node2ind, node3ind = elements[ind,0], elements[ind,1], elements[ind,2], elements[ind,3]
        ele_list.append(fea.ChElementTetra_4())
        ele_list[ind].SetNodes(node_list[node0ind], node_list[node1ind], node_list[node2ind], node_list[node3ind])
        ele_list[ind].SetMaterial(material)
        mesh.AddElement(ele_list[ind]) # use 0-based indexing here
        
    mesh.SetAutomaticGravity(False)
    system.Add(mesh)
    
    zmax = np.max(nodes,axis = 0)[2] - plateThickness
    zmin = np.min(nodes,axis = 0)[2] + plateThickness
    
    faceA_nodeind = np.asarray(np.nonzero(nodes[:,2]>zmax))[0]
    faceB_nodeind = np.asarray(np.nonzero(nodes[:,2]<zmin))[0]
    
    # Distribute total force over nodes on face A
    force_dist = force_total / len(faceA_nodeind)
    
    for ind in faceA_nodeind:
        node_list[ind].SetForce(chrono.ChVectorD(0,0,-force_dist))
    
    # Face B: Fixed truss
    truss = chrono.ChBody()
    truss.SetBodyFixed(True)
    system.Add(truss)
    
    # Face A: Moving truss
    truss_moving = chrono.ChBody()
    system.Add(truss_moving)
    
    # Add face B nodes to fixed truss
    constr_list = []
    for ind in faceB_nodeind:
        constr_list.append(fea.ChLinkPointFrame())
        constr_list[-1].Initialize(node_list[ind],truss)
        system.Add(constr_list[-1])
    
    # Add face A nodes to moving truss
    constr_list_moving = []
    for ind in faceA_nodeind:
        constr_list_moving.append(fea.ChLinkPointFrame())
        constr_list_moving[-1].Initialize(node_list[ind],truss_moving)
        system.Add(constr_list_moving[-1])
    
    # Prismatic Joint, this works for Paradiso MKL solver
    constr_A = chrono.ChLinkLockPrismatic()
    constr_A.Initialize(truss, truss_moving, chrono.ChCoordsysD())
    system.AddLink(constr_A)
    
    if solver == "ParadisoMKL": # TODO: Validate results from this solver
        msolver = mkl.ChSolverPardisoMKL()
        msolver.LockSparsityPattern(True)
    else: # MINRES solver
        msolver = chrono.ChSolverMINRES()
        msolver.SetMaxIterations(100)
        msolver.SetTolerance(1e-10)
        msolver.EnableDiagonalPreconditioner(True)
        msolver.SetVerbose(True)
    
    system.SetSolver(msolver)
    system.DoStaticLinear()
    
    ## Format FEA output
    
    # Retrieve nodal position
    node_arr_sol = np.copy(nodes) ###### <<< nodal solution
    for ind in range(nodes.shape[0]):
        pos = node_list[ind].GetPos()
        node_arr_sol[ind,0] = pos.x
        node_arr_sol[ind,1] = pos.y
        node_arr_sol[ind,2] = pos.z
    
    # element centroid positions
    ele_centroid = np.empty([len(ele_list),3]) ##### <<< elemental centroid position
    for ind in range(len(ele_list)):
        sumxyz = np.zeros([1,3]) # [x,y,z] for centroid
        for iind in range(4): # iterate over 4 nodes
            nodeind = elements[ind,iind] # all zero-based indices now
            sumxyz += nodes[nodeind,0:3]
        sumxyz = sumxyz / 4 # center of 4 nodes
        ele_centroid[ind,:] = sumxyz
        
    ele_Tstress = np.empty((len(ele_list),6))
    ele_Tstrain = np.empty((len(ele_list),6))
    ele_VMstress = np.empty((len(ele_list),1))
    
    for ind in range(len(ele_list)):
        
        ep = ele_list[ind].GetStrain()
        sig = ele_list[ind].GetStress()
        sigVM = sig.GetEquivalentVonMises()
    
        strain_vector = np.array([ep.XX(), ep.YY(), ep.ZZ(), ep.XY(), ep.XZ(), ep.YZ()])
        stress_vector = np.array([sig.XX(), sig.YY(), sig.ZZ(), sig.XY(), sig.XZ(), sig.YZ()])
    
        ele_Tstress[ind,:] = stress_vector.reshape((1,6))
        ele_Tstrain[ind,:] = strain_vector.reshape((1,6))
        ele_VMstress[ind,:] = sigVM
        
    feaResult = {
        "nodes" : nodes,
        "displacement" : node_arr_sol - nodes,
        "elementCentroids" : ele_centroid,
        "nodeIndA" : faceA_nodeind,
        "nodeIndB" : faceB_nodeind,
        "elementVMstresses" : ele_VMstress,
        "elementStresses" : ele_Tstress,
        "elementStrains" : ele_Tstrain,
        "force" : force_total
        }
    
    return feaResult

def computeFEAElasticModulus(feaResult):
    # Compute elastic modulus from feaResult
    
    displacement = feaResult["displacement"]
    nodeIndA = feaResult["nodeIndA"]
    force = feaResult["force"]
    displacementAinZ = np.mean(displacement[nodeIndA,2])
    
    return force / displacementAinZ

def saveSurfaceMesh(filename, vertices, faces):
    # Save surface mesh (tested on .STL only)
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces.copy())
    mesh.export(filename)

def saveTetrahedralMesh(filename, tet):
    pv.save_meshio(filename, tet.grid)

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import nrrd
    
    voxelSize = (0.05, 0.05, 0.05) # mm
    plattenThicknessVoxels = 10 # voxels
    plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm
    cubeShape = (201, 201, 201)
    
    filenameNRRD = "../../data/rois/isodata_04216_roi_4.nrrd"
    filenameSTL = "../../data/output/isodata_04216_roi_4.stl"
    filenameVTK = "../../data/output/isodata_04216_roi_4.vtk"
    
    # Elastic Modulus of a real bone ROI
    roiBone, header = nrrd.read(filenameNRRD)
    roiBone = addPlatten(roiBone, plattenThicknessVoxels)
    vertices, faces, normals, values = Voxel2SurfMesh(roiBone, voxelSize=(0.05,0.05,0.05), step_size=2)
    saveSurfaceMesh(filenameSTL, vertices, faces)
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    nodes, elements, tet = Surf2TetMesh(vertices, faces, verbose=0)
    saveTetrahedralMesh(filenameVTK, tet)
    feaResult = computeFEACompressLinear(nodes, elements, plattenThicknessMM, solver="ParadisoMKL")
    elasticModulus = computeFEAElasticModulus(feaResult)
    print(elasticModulus)
    
    # Elastic Modulus of solid chunk of bone
    roiCube = np.ones(cubeShape).astype(bool)
    roiCube[0,:,:] = False; roiCube[-1,:,:] = False
    roiCube[:,0,:] = False; roiCube[:,-1,:] = False
    roiCube[:,:,0] = False; roiCube[:,:,-1] = False
    vertices, faces, normals, values = Voxel2SurfMesh(roiCube, voxelSize=(0.05,0.05,0.05), step_size=2)
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    nodes, elements, tet = Surf2TetMesh(vertices, faces, verbose=0)
    feaResult0 = computeFEACompressLinear(nodes, elements, plattenThicknessMM, solver="ParadisoMKL")
    elasticModulus0 = computeFEAElasticModulus(feaResult0)
    print(elasticModulus0)
    
    f = feaResult
    plt.plot(f['nodes'][:,1],f['nodes'][:,2],'ko')
    plt.plot(f['nodes'][:,1]+f['displacement'][:,1]*10e9,
              f['nodes'][:,2]+f['displacement'][:,2]*10e9,'ro')