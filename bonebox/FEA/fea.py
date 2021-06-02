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

# finite element library
import pychrono as chrono
import pychrono.fea as fea
# import pychrono.mkl as mkl

def addPlatten(volume, plattenThicknessVoxels):
    # adds compression plates in Z
    
    vmax = np.max(volume)
    volume[:,:,0:plattenThicknessVoxels] = vmax
    volume[:,:,-plattenThicknessVoxels:-1] = vmax
    
    return volume

def Voxel2HexaMesh(volume, voxelSize=(1,1,1), origin=None):
    # TODO: Directly convert voxels to hexamesh
    # origin: (0,0,0) corresponds to "top left corner", defaults to None (center of volume)
    pass

def Voxel2SurfMesh(volume, voxelSize=(1,1,1), origin=None, level=None, step_size=1, allow_degenerate=False):
    # Convert voxel image to surface
    
    if level == None:
        level = (np.max(volume)+np.min(volume))/2
    
    vertices, faces, normals, values = \
        measure.marching_cubes_lewiner(volume = volume, level = level, spacing = voxelSize, \
                                       step_size = step_size, allow_degenerate = allow_degenerate)
    return vertices, faces, normals, values

def Surf2TetMesh(vertices, faces, order=1, verbose=1):
    # Convert surface mesh to tetrahedra
    
    tet = tetgen.TetGen(vertices,faces)
    tet.tetrahedralize(order=1,verbose=1)
    
    nodes = tet.node
    elements = tet.elem
    
    return nodes, elements

def isWatertight(vertices, faces):
    # Check if mesh is watertight
    mesh = trimesh.Trimesh(vertices = vertices, faces = faces)
    return mesh.is_watertight
    
def computeFEACompressLinear(nodes, elements, plateThickness, \
                             elasticModulus=17e9, poissonRatio=0.3, \
                             force_total = 1):
    # Linear Finite Element Analysis with PyChrono
    # plateThickness: Thickness of compression plates in absolute units

    system = chrono.ChSystemNSC()
    material = chrono.fea.ChContinuumElastic(elasticModulus, poissonRatio)
    mesh = fea.ChMesh()

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
        
    truss = chrono.ChBody()
    truss.SetBodyFixed(True)
    system.Add(truss)
    
    truss_moving = chrono.ChBody()
    system.Add(truss_moving)
    
    constr_list = []
    for ind in faceB_nodeind:
        constr_list.append(fea.ChLinkPointFrame())
        constr_list[-1].Initialize(node_list[ind],truss)
        system.Add(constr_list[-1])
    
    # Create a truss for the moving face
    constr_list_moving = []
    for ind in faceA_nodeind:
        constr_list_moving.append(fea.ChLinkPointFrame())
        constr_list_moving[-1].Initialize(node_list[ind],truss_moving)
        system.Add(constr_list_moving[-1])
        
    # msolver = mkl.ChSolverMKL()
    # msolver.LockSparsityPattern(True)

    # MINRES solver
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

if __name__ == "__main__":
    
    voxelSize = (0.05, 0.05, 0.05) # mm
    plattenThicknessVoxels = 10 # voxels
    plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm
    
    roiBone, header = nrrd.read("../../data/rois/isodata_04216_roi_4.nrrd")
    
    roiBone = addPlatten(roiBone, plattenThicknessVoxels)
    vertices, faces, normals, values = Voxel2SurfMesh(roiBone, voxelSize=(0.05,0.05,0.05))
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    nodes, elements = Surf2TetMesh(vertices, faces)
    feaResult = computeFEACompressLinear(nodes, elements, plattenThicknessMM)
    elasticModulus = computeFEAElasticModulus(feaResult)