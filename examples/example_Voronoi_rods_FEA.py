# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:55:11 2021

@author: Qian.Cao

Generate a series of Voronoi Rod phantoms

"""

import sys
sys.path.append('../') # use bonebox from source without having to install/build

from bonebox.phantoms.TrabeculaeVoronoi import *
import numpy as np
import matplotlib.pyplot as plt
# import mcubes

from bonebox.FEA.fea import *

plt.ion()

print('Running example for TrabeculaeVoronoi')

def makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState):

    # Parameters for generating phantom mesh
    Sxyz, Nxyz = (5,5,5), (Nseeds, Nseeds, Nseeds) # volume extent in XYZ (mm), number of seeds along XYZ
    Rxyz = 1.
    # edgesRetainFraction = 0.5
    facesRetainFraction = 0.5
    # dilationRadius = 3 # (voxels)
    # randState = 123 # for repeatability
    
    # Parameters for generating phantom volume
    volumeSizeVoxels = (100,100,100)
    voxelSize = np.array(Sxyz) / np.array(volumeSizeVoxels)
    
    # Generate faces and edges
    points = makeSeedPointsCartesian(Sxyz, Nxyz)
    ppoints = perturbSeedPointsCartesianUniformXYZ(points, Rxyz, randState=randState)
    vor, ind = applyVoronoi(ppoints, Sxyz)
    uniqueEdges, uniqueFaces = findUniqueEdgesAndFaces(vor, ind)
    
    # Compute edge cosines
    edgeVertices = getEdgeVertices(vor.vertices, uniqueEdges)
    edgeCosines = computeEdgeCosine(edgeVertices, direction = (0,0,1))
    
    # Compute face properties
    faceVertices = getFaceVertices(vor.vertices, uniqueFaces)
    faceAreas = computeFaceAreas(faceVertices)
    faceCentroids = computeFaceCentroids(faceVertices)
    faceNormas = computeFaceNormals(faceVertices)
    
    # Filter random edges and faces
    uniqueEdgesRetain, edgesRetainInd = filterEdgesRandomUniform(uniqueEdges, 
                                                                 edgesRetainFraction, 
                                                                 randState=randState)
    uniqueFacesRetain, facesRetainInd = filterFacesRandomUniform(uniqueFaces, 
                                                                 facesRetainFraction, 
                                                                 randState=randState)
    
    volume = makeSkeletonVolumeEdges(vor.vertices, uniqueEdgesRetain, voxelSize, volumeSizeVoxels)
    volumeDilated = dilateVolumeSphereUniform(volume, dilationRadius)
    
    volumeDilated = volumeDilated.astype(float)
    bvtv = np.sum(volumeDilated>0) / volume.size
    
    edgeVerticesRetain = getEdgeVertices(vor.vertices, uniqueEdgesRetain)
    
    volumeDilated = setEdgesZero(volumeDilated)
    
    return volumeDilated, bvtv, edgeVerticesRetain
    
    # Visualize all edges
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for ii in range(edgeVertices.shape[0]):
    #     ax.plot(edgeVertices[ii,:,0],edgeVertices[ii,:,1],edgeVertices[ii,:,2],'b-')
    
    # volumeSmoothed = mcubes.smooth(volumeDilated)

rhoBone = 2e-3 # g/mm3
voxelSize = (0.05, 0.05, 0.05) # mm
pixelSize = (0.05, 0.05) # mm
radiusTBS = 5 # pixels
plattenThicknessVoxels = 5 # voxels
plattenThicknessMM = plattenThicknessVoxels * voxelSize[0] # mm    

def computeFEA(volume):

    roiBone = addPlatten(volume, plattenThicknessVoxels)
    vertices, faces, normals, values = Voxel2SurfMesh(roiBone, voxelSize=(0.05,0.05,0.05), step_size=1)
    print("Is watertight? " + str(isWatertight(vertices, faces)))
    nodes, elements, tet = Surf2TetMesh(vertices, faces, verbose=0)
    feaResult = computeFEACompressLinear(nodes, elements, plattenThicknessMM)
    elasticModulus = computeFEAElasticModulus(feaResult)
    
    return elasticModulus

#%%

# Projection/TBS Settings
dilationRadius = 3.4
all_randState = [1,2,3]
all_Nseeds = np.arange(8,21).astype(int)
all_edgesRetainFraction = np.linspace(0.4,0.6,11)

bvtvs = np.zeros((len(all_randState),len(all_Nseeds),len(all_edgesRetainFraction)))
Es = np.zeros(bvtvs.shape)

for rr, randState in enumerate(all_randState):
    for nn, Nseeds in enumerate(all_Nseeds):
        for ee, edgesRetainFraction in enumerate(all_edgesRetainFraction):
            
            print(rr,nn,ee)

            volume, bvtv, edgeVerticesRetain = makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState)
            E = computeFEA(volume)
            
            bvtvs[rr,nn,ee] = bvtv
            Es[rr,nn,ee] = E
            
#%%

np.save("FEA_bvtvs", bvtvs)
np.save("FEA_Es", Es)

# # np.save("FEAbvtvs",bvtvs)
# # np.save("FEAEs",Es)

# plt.imshow(np.mean(bvtvs,axis=0))
# plt.axis("off")
# plt.colorbar()

# plt.imshow(np.std(bvtvs,axis=0))
# plt.axis("off")
# plt.colorbar()

# plt.imshow(np.mean(Es,axis=0))
# plt.axis("off")
# plt.colorbar()

# plt.imshow(np.std(Es,axis=0))
# plt.axis("off")
# plt.colorbar()

# plt.plot(bvtvs.flatten(),Es.flatten(),'ko')