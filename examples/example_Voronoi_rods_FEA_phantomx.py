# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:55:11 2021

@author: Qian.Cao

Generate a series of Voronoi Rod (AND PLATE) phantoms

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

#% Generate Phantoms with different dilation Radii and seeds

out_dir = "/data/BoneBox-out/"
phantoms_dir = "/data/BoneBox-out/phantoms/"

# Projection/TBS Settings
all_randState = [1,2,3,4]
all_dilationRadius = np.linspace(1,3,10)
all_Nseeds = np.arange(3,10).astype(int)
edgesRetainFraction = 0.8

bvtvs = np.zeros((len(all_randState),len(all_Nseeds),len(all_dilationRadius)))
# Es = np.zeros(bvtvs.shape)

#% Load Existing BvTv

bvtvs = np.load(out_dir+"FEA_bvtvs_2.npy")

import scipy.ndimage

# plot mean and Std of bvtvs
meanbvtv = scipy.ndimage.zoom(np.mean(bvtvs,axis=0), 10, order=1)
extent=[np.min(all_dilationRadius),np.max(all_dilationRadius),
        np.min(all_Nseeds),np.max(all_Nseeds)]

fig, ax = plt.subplots()
ax.imshow(meanbvtv, interpolation="nearest",extent=extent,aspect='auto', origin='lower')
cs = ax.contour(meanbvtv, extent=extent, colors='w', origin='lower')
ax.clabel(cs, fontsize=9, inline=True)

plt.xlabel("Dilation Radius")
plt.ylabel("N")

# Get contour lines
samplesPerVal = 7
isobvtv_vals = [0.24, 0.32, 0.4]
isobvtv_xy = []
isobvtv_xyeval = []

for vv, isobvtv in enumerate(isobvtv_vals):
    cs = ax.contour(meanbvtv, [isobvtv], extent=extent)
    p = cs.collections[0].get_paths()[0]
    isobvtv_xy.append(p.vertices)
    
for ii, xy in enumerate(isobvtv_xy):
    
    xylist = []
    
    Nxy = xy.shape[0]
    Dxy = Nxy // (samplesPerVal-1)
    
    # select 4 evenly spaced points along the isocline
    for ss in range(samplesPerVal-1):
        xylist.append(xy[ss*Dxy,:])
    xylist.append(xy[-1,:])
    
    isobvtv_xyeval.append(np.array(xylist))
    
#% Phantom HU

# Parameters for PhantomX
mmPerInch = 25.4

HUBone = 1500
HUAir = -800
HUMarrow = -50

dimXcm = 27
dimYcm = 19

DPIxy = 300
voxXY = 1/(DPIxy/mmPerInch)
voxZ = 0.071428571 # mm

dimX = int(270//voxXY)
dimY = int(190//voxXY)
dimZ = 100

phantomsInX = dimX // 150 # 21 {3 BvTv} x {7 Samples}
phantomsInY = dimY // 150 # 14 RandStates

phanHU = np.ones((dimX,dimY,dimZ))*(-800)

randStates = np.arange(14)

#%%

for ii, isoval in enumerate(isobvtv_vals): # 3
    
    dilationRadiusArray = isobvtv_xyeval[ii][:,0]
    NseedsArray = isobvtv_xyeval[ii][:,1].astype(int)
    
    for pp in range(samplesPerVal): # 7
        
        for rr, randState in enumerate(randStates): # 14
            
            print(ii, pp, rr)
            
            xind = (ii*7+pp)*(dimX // phantomsInX)
            yind = rr*(dimY // phantomsInY)
            
            volume, bvtv, edgeVerticesRetain = makePhantom(dilationRadiusArray[pp],
                                                            NseedsArray[pp],
                                                            edgesRetainFraction,
                                                            randState)
            
            # Convert to HU
            volume[volume==1] = HUBone
            volume[volume==0] = HUMarrow
            
            # Assign to phanHU
            phanHU[xind:xind+100,yind:yind+100,:] = volume
            
            # np.save(phantoms_iso_dir+"phan_"+str(ii)+"_"+str(pp)+"_"+str(rr), volume)
            # plt.subplot(1, 3, 1)
            # plt.imshow(volume[:,:,50],interpolation="nearest",cmap="gray")
            # plt.title("XY"); plt.axis("off")
            # plt.subplot(1, 3, 2)
            # plt.imshow(volume[:,50,:],interpolation="nearest",cmap="gray")
            # plt.title("XZ"); plt.axis("off")
            # plt.subplot(1, 3, 3)
            # plt.imshow(volume[50,:,:],interpolation="nearest",cmap="gray")
            # plt.title("YZ"); plt.axis("off")
            # plt.savefig(phantoms_iso_dir+"fig_"+str(ii)+"_"+str(pp)+"_"+str(rr))
            # plt.close("all")
            
            # E = computeFEA(volume)
            # EsArr[ii,pp,rr] = E
            
            # print(E)
            
np.save(out_dir+"PhantomX_20210623", phanHU)

#%%

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(phanHU[:,:,50].T,cmap="gray")
plt.axis("off")
plt.savefig(out_dir+'PhantomX_20210623.png',bbox_inches = 'tight',
    pad_inches = 0)

#%% Shuffle - Load File

phanHU = np.load(out_dir+"PhantomX_20210623.npy")

phanHU = np.delete(phanHU,obj=np.arange(3171,3188),axis=0)

#%% Shuffle - Roll Matrix Positions

for rr, randState in enumerate(randStates):
    
    yind = rr*(dimY // phantomsInY)
    
    phanHU[:,yind:yind+100,:] = np.roll(phanHU[:,yind:yind+100,:], shift=rr*7*(dimX // phantomsInX), axis=0)
    
np.save(out_dir+"PhantomX_20210623_shuffle.npy", phanHU)

#%% Condense into smaller "batches"

phanHUsh = np.load(out_dir+"PhantomX_20210623_shuffle.npy")

#%% Numpy to DICOM

phanHUsh = np.load(out_dir+"PhantomX_20210623_shuffle.npy")

data = phanHUsh.astype(np.int32)

import SimpleITK as sitk

image = sitk.GetImageFromArray(data)

#%%

xinds = []
yinds = []

for ii, isoval in enumerate(isobvtv_vals): # 3
    
    for pp in range(samplesPerVal): # 7
        
        for rr, randState in enumerate(randStates): # 14
            
            print(ii, pp, rr)
            
            xind = (ii*7+pp)*(dimX // phantomsInX)
            yind = rr*(dimY // phantomsInY)
            
            xinds.append(xind)
            yinds.append(yind)
            
yinds = np.array(yinds)[:14]
xinds = np.array(xinds)




#%% save shuffled image

import matplotlib.pyplot as plt
import numpy as np
plt.imshow(phanHU[:,:,50].T,cmap="gray")
plt.axis("off")
plt.savefig(out_dir+'PhantomX_20210623_shuffle.png',bbox_inches = 'tight',
    pad_inches = 0)

#%%

from PIL import Image
import numpy as np

data = phanHU[:,:,50]

data[data==1500] = 255
data[data==-50] = 50
data[data==-800] = 0

data = data.astype(np.uint8)

#Rescale to 0-255 and convert to uint8
# rescaled = (255.0 / data.max() * (data - (data.min()))).astype(np.uint8)

im = Image.fromarray(rescaled)
im.save(out_dir+'PhantomX_20210623.png')

# for xx in range(phantomsInX):
#     for yy in range(phantomsInY):

#%%

# for rr, randState in enumerate(all_randState):
#     for nn, Nseeds in enumerate(all_Nseeds):
#         for ra, dilationRadius in enumerate(all_dilationRadius):
            
#             print(rr,nn,ra)

#             volume, bvtv, edgeVerticesRetain = makePhantom(dilationRadius, Nseeds, edgesRetainFraction, randState)
#             np.save(phantoms_dir+"phan_"+str(rr)+"_"+str(nn)+"_"+str(ra), volume)
#             # E = computeFEA(volume)
            
#             bvtvs[rr,nn,ra] = bvtv
#             # Es[rr,nn,ra] = E
            
# np.save(out_dir+"FEA_bvtvs_2", bvtvs)
# # np.save("FEA_Es_2", Es)

#%%

# bvtvs = np.load(out_dir+"FEA_bvtvs_2.npy")

# import scipy.ndimage

# # plot mean and Std of bvtvs
# meanbvtv = scipy.ndimage.zoom(np.mean(bvtvs,axis=0), 10, order=1)
# extent=[np.min(all_dilationRadius),np.max(all_dilationRadius),
#         np.min(all_Nseeds),np.max(all_Nseeds)]

# fig, ax = plt.subplots()
# ax.imshow(meanbvtv, interpolation="nearest",extent=extent,aspect='auto', origin='lower')
# cs = ax.contour(meanbvtv, extent=extent, colors='w', origin='lower')
# ax.clabel(cs, fontsize=9, inline=True)

# plt.xlabel("Dilation Radius")
# plt.ylabel("N")

# # Get contour lines
# samplesPerVal = 4
# isobvtv_vals = [0.32]
# isobvtv_xy = []
# isobvtv_xyeval = []

# for vv, isobvtv in enumerate(isobvtv_vals):
#     cs = ax.contour(meanbvtv, [isobvtv], extent=extent)
#     p = cs.collections[0].get_paths()[0]
#     isobvtv_xy.append(p.vertices)
    
# for ii, xy in enumerate(isobvtv_xy):
    
#     xylist = []
    
#     Nxy = xy.shape[0]
#     Dxy = Nxy // (samplesPerVal-1)
    
#     # select 4 evenly spaced points along the isocline
#     for ss in range(samplesPerVal-1):
#         xylist.append(xy[ss*Dxy,:])
#     xylist.append(xy[-1,:])
    
#     isobvtv_xyeval.append(np.array(xylist))
    
#%%

# #%% Export Triplanar images of the phantoms

# for rr, randState in enumerate(all_randState):
#     for nn, Nseeds in enumerate(all_Nseeds):
#         for ra, dilationRadius in enumerate(all_dilationRadius):
            
#             volume = np.load(phantoms_dir+"phan_"+str(rr)+"_"+str(nn)+"_"+str(ra)+".npy")
#             plt.subplot(1, 3, 1)
#             plt.imshow(volume[:,:,50],interpolation="nearest",cmap="gray")
#             plt.title("XY"); plt.axis("off")
#             plt.subplot(1, 3, 2)
#             plt.imshow(volume[:,50,:],interpolation="nearest",cmap="gray")
#             plt.title("XZ"); plt.axis("off")
#             plt.subplot(1, 3, 3)
#             plt.imshow(volume[50,:,:],interpolation="nearest",cmap="gray")
#             plt.title("YZ"); plt.axis("off")
            
#             plt.savefig(phantoms_dir+"fig_"+str(rr)+"_"+str(nn)+"_"+str(ra))
#             plt.close("all")

# #%% Evaluates points along isobvtv_xyeval

# phantoms_iso_dir = "/data/BoneBox-out/phantoms_iso/"

# randStates = [2, 3]
# EsArr= np.zeros((len(isobvtv_vals),samplesPerVal,len(randStates)))

# for ii, isoval in enumerate(isobvtv_vals):
    
#     dilationRadiusArray = isobvtv_xyeval[ii][:,0]
#     NseedsArray = isobvtv_xyeval[ii][:,1].astype(int)
    
#     for pp in range(samplesPerVal):
#         for rr, randState in enumerate(randStates):
            
#             print(ii, pp, rr)
            
#             volume, bvtv, edgeVerticesRetain = makePhantom(dilationRadiusArray[pp],
#                                                            NseedsArray[pp],
#                                                            edgesRetainFraction,
#                                                            randState)
            
#             np.save(phantoms_iso_dir+"phan_"+str(ii)+"_"+str(pp)+"_"+str(rr), volume)
#             plt.subplot(1, 3, 1)
#             plt.imshow(volume[:,:,50],interpolation="nearest",cmap="gray")
#             plt.title("XY"); plt.axis("off")
#             plt.subplot(1, 3, 2)
#             plt.imshow(volume[:,50,:],interpolation="nearest",cmap="gray")
#             plt.title("XZ"); plt.axis("off")
#             plt.subplot(1, 3, 3)
#             plt.imshow(volume[50,:,:],interpolation="nearest",cmap="gray")
#             plt.title("YZ"); plt.axis("off")
#             plt.savefig(phantoms_iso_dir+"fig_"+str(ii)+"_"+str(pp)+"_"+str(rr))
#             plt.close("all")
            
#             E = computeFEA(volume)
#             EsArr[ii,pp,rr] = E
            
#             print(E)
            
# np.save(out_dir+"EsArr", EsArr)

# #%%



# #%%

# vol = np.ones(volume.shape).astype(bool)
# vol[0,:,:] = False; vol[-1,:,:] = False; 
# vol[:,0,:] = False; vol[:,-1,:] = False; 
# vol[:,:,0] = False; vol[:,:,-1] = False; 

# E0 = computeFEA(vol)

# #%%

# a_strings = ["%.2f" % x for x in dilationRadiusArray]
# plt.boxplot(np.squeeze(EsArr).T/E0)
# plt.xticks([1, 2, 3, 4], a_strings)
# plt.grid()

#%%

# EsArray = np.array(EsList)

# bvtvs = np.load(out_dir+"FEA_bvtvs_2.npy")

# Es = np.load("FEA_Es_2.npy")

# plt.plot(bvtvs.flatten(), -Es.flatten(),'ko')

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