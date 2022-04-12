#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 09:43:49 2022

@author: sriharsha.marupudi
"""

import skimage 
from skimage import morphology
import sys
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
from collections import defaultdict
from pylab import cross,dot,inv
from scipy import ndimage
from glob import glob

import os
    
NAME = "143-756"

segmentationsDir = "/gpfs_projects/sriharsha.marupudi/Segmentations/"
normalsTXT = glob(segmentationsDir+"Normals-*.txt")

outDir = "/gpfs_projects/sriharsha.marupudi/extract_rois_output/"

print(f"Previewing normal vectors for sample {NAME}.")

normVec = lambda v: v / np.sqrt(np.sum(v**2, axis=1))[:,None]

data = defaultdict(list)
readType = None
with open(f"{segmentationsDir}/Normals-{NAME}.txt") as file:
    for line in file:
        
        line = line.rstrip()
        
        if "#" in line:
            discard = line.find("#")
            line = line[:discard]
        
        if "3D" in line:
            readType = "normals"
            
        if "Position" in line:
            readType = "positions"
            
        if len(line) == 0:
            readType = None
            
        if readType is not None and "," in line:
            data[readType].append([float(x) for x in line.split(",")[:3]])

seg, header = nrrd.read(f"{segmentationsDir}/Segmentation-{NAME}.seg.nrrd")
plt.imshow(seg[:,:,450])
plt.show()
exc,head = nrrd.read(f"{segmentationsDir}/Segmentation-Exclude-{NAME}.seg.nrrd")

x = np.logical_not(exc)#Use for segmentation that has an exclusion map 
seg_exc = np.logical_and(seg,x)#Use for segmentation that has an exclusion map 
#seg_exc = np.logical_and(seg,exc) #Use for segmentation that has an exclusion map 
#seg_close_exc = ndimage.morphology.binary_closing(seg_exc) # Use if there is an exclusion map for this segmentation
#seg_close_exc = ndimage.morphology.binary_closing(seg) #Use if there is no exclusion map for this segmentation 
seg_close_exc = ndimage.morphology.binary_closing(seg_exc[:,:,450],np.ones((200,200)))
#plt.imshow(seg_close_exc);plt.show()

#Load exclusion map to crop 
#seg_close_erode = ndimage.morphology.binary_erosion(seg_close_exc)
#seg_close_erode = ndimage.morphology.binary_erosion(seg_close_exc,np.ones((50,50)))
#plt.imshow(seg_close_exc+seg_close_erode);plt.show()
#plt.imshow(seg_close_exc.astype(int)+seg_close_erode.astype(int)+seg[:,:,450]);plt.show()
#seg_close_erode = ndimage.morphology.binary_erosion(seg_close_exc)
#erosion = np.nonzero(seg_close_erode)
seg_close_erode = ndimage.morphology.binary_erosion(seg_close_exc,np.ones((300,300)))
erosion = np.nonzero(seg_close_erode)
#plt.imshow(seg_close_exc.astype(int)+seg_close_erode.astype(int)+seg[:,:,450]);plt.show()

# normalize normal vectors to 1
normals = normVec(np.array(data["normals"]))



vecind = np.arange(len(data["normals"]))
meanPositions = []
for ind in vecind:
    positions = data["positions"][ind:len(data["positions"]):len(data["normals"])]
    meanPositions.append(np.mean(positions,axis=0))
    
Nbones = round(len(data["normals"])/2) # two normal vectors per bone
    
#%%

sliceImg = []
sliceImg.append(lambda img, x: img[int(x),:,:])
sliceImg.append(lambda img, y: img[:,int(y),:])
sliceImg.append(lambda img, z: img[:,:,int(z)])

nl = 300 # normal vector length in the plot

def plotNormals(ax,img,pos,norm,xyz):
    axes = "XYZ"
    inds = [0,1,2]
    ind = inds.pop(axes.find(xyz)) # this is the index
    ax.imshow(sliceImg[ind](seg, int(pos[ind])), cmap="gray")
    ax.plot(pos[inds[-1]], pos[inds[-2]], 'yo', markersize=10,markerfacecolor='none')
    ax.arrow(pos[inds[-1]], pos[inds[-2]], nl*norm[inds[-1]], nl*norm[inds[-2]], color="y", linewidth=1)
    ax.set_title(f"{xyz} Slice")
    ax.set_xlabel(axes[inds[-1]])
    ax.set_ylabel(axes[inds[-2]])

for boneInd in range(Nbones):

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3,figsize=(15,30))
    fig.suptitle(f"Segmentation {NAME}, bone {boneInd}")
    
    plotNormals(ax1,seg,meanPositions[2*boneInd],normals[2*boneInd],"X")
    plotNormals(ax2,seg,meanPositions[2*boneInd],normals[2*boneInd],"Y")
    plotNormals(ax3,seg,meanPositions[2*boneInd],normals[2*boneInd],"Z")
    
    plotNormals(ax4,seg,meanPositions[2*boneInd+1],normals[2*boneInd+1],"X")
    plotNormals(ax5,seg,meanPositions[2*boneInd+1],normals[2*boneInd+1],"Y")
    plotNormals(ax6,seg,meanPositions[2*boneInd+1],normals[2*boneInd+1],"Z")
    
    # plt.tight_layout()
    
#%% Extract an ROI at the centroid
    


roi_dims = (300,300,300) # size of my ROI
roi_centroid = np.mean(meanPositions,axis=0).round().astype(int)

xx, yy, zz = np.meshgrid(np.linspace(-150,149,300), np.linspace(-150,149,300), np.linspace(-150,149,300),indexing = "ij")

#index = ij to plot function using row,col rather then x,y order 
points = xxt, yyt, zzt = xx.copy(), yy.copy(), zz.copy()

Q = np.array([0,0,1])
U = normals[0]
V = normals[1]
W = normals[2]
X = normals[3]
Y = normals[4]
Z = normals[5]


def rotation_matrix_from_vectors(U, V):

    a, b = (U), (V)
    v = np.cross(a,b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
    #np.save("/gpfs_projects/sriharsha.marupudi/extract_rois_output/rotation_matrix-{NAME}.npy",rotation_matrix)
    #save rotation matrix 
rot = rotation_matrix_from_vectors(U,V)
np.save(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/rotation_matrix_{NAME}.npy",rot)
rotation_result = np.matmul(Q,rot)
print(rotation_result)

points_array = np.asarray(points)
points_flatten = points_array.flatten()
a = points_flatten.shape
b =a[0]
c = b//3
points_array1 = points_flatten.reshape(3,c)
#points_array1 = points_flatten.reshape(3,8000000)
sample = rot.dot(points_array1)
np.save(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/transform_coordinates_{NAME}.npy",sample)
#save as numpy array

# Step 1: compute rotation matrix from (0,0,1) to normals[0] and normals[1]
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
# https://stackoverflow.com/questions/54445195/np-tensordot-for-rotation-of-point-clouds
# apply rotation matrix to each point: (xxt[i], yyt[i], zzt[i])

# Step 2: Shift origin from (0,0,0) to roi_centroid
#np.nonzero index results to shift 

erosion = np.nonzero(seg_close_erode)

s = erosion[1]
shift = s.flatten()

#shift0 = erosion[0]
#shift1 = erosion[1]

#roi1
sample1 = sample[0] + roi_centroid[0]
sample2 = sample[1] + roi_centroid[1]
sample3 = sample[2] + roi_centroid[2]
#roi2
sample4 = sample[0] + shift[0]
sample5 = sample[1] + shift[1]
sample6 = sample[2] + shift[2]
#roi3
sample7 = sample[0] + shift[3]
sample8 = sample[1] + shift[4]
sample9 =sample[2] + shift[5]
#roi4
sample10= sample[0] + shift[10]
sample11= sample[1] + shift[11]
sample12= sample[2] + shift[12]
#roi5
sample13= sample[0] + shift[100]
sample14= sample[1] + shift[200]
sample15= sample[2] + shift[300]

#adding offsets for multiple roi 
#Points from erosion 
roi1 = ndimage.map_coordinates(seg,[sample1.flatten(), sample2.flatten(), sample3.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi1[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")

roi2 = ndimage.map_coordinates(seg,[sample4.flatten(), sample5.flatten(), sample6.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi2[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi3 = ndimage.map_coordinates(seg,[sample7.flatten(), sample8.flatten(), sample9.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi3[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi4 = ndimage.map_coordinates(seg,[sample10.flatten(), sample11.flatten(), sample12.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi4[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi5= ndimage.map_coordinates(seg,[sample13.flatten(), sample14.flatten(), sample15.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi5[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 

#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/ROI-1_{NAME}.nrrd",roi1)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/ROI-2_{NAME}.nrrd",roi2)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/ROI-3_{NAME}.nrrd",roi3)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/ROI-4_{NAME}.nrrd",roi4)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/ROI-5_{NAME}.nrrd",roi5)

#bone2
def rotation_matrix_from_vectors(W, X):

    a2, b2 = (W), (X)
    v = np.cross(a2,b2)
    c = np.dot(a2, b2)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix2 = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix2
    #np.save("/gpfs_projects/sriharsha.marupudi/extract_rois_output/rotation_matrix-{NAME}.npy",rotation_matrix)
    #save rotation matrix 
rot2 = rotation_matrix_from_vectors(W,X)
np.save(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/rotation_matrix2_{NAME}.npy",rot)
rotation_result2 = np.matmul(Q,rot2)
print(rotation_result2)

points_array = np.asarray(points)
points_flatten = points_array.flatten()
a = points_flatten.shape
b =a[0]
c = b//3
points_array1 = points_flatten.reshape(3,c)
#points_array1 = points_flatten.reshape(3,8000000)
sample2 = rot2.dot(points_array1)
np.save(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/transform_coordinates2_{NAME}.npy",sample)
#save as numpy array

# Step 1: compute rotation matrix from (0,0,1) to normals[0] and normals[1]
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
# https://stackoverflow.com/questions/54445195/np-tensordot-for-rotation-of-point-clouds
# apply rotation matrix to each point: (xxt[i], yyt[i], zzt[i])

# Step 2: Shift origin from (0,0,0) to roi_centroid
#np.nonzero index results to shift 

erosion2 = np.nonzero(seg_close_erode)

s2 = erosion2[1]
shift2 = s2.flatten()

#shift0 = erosion[0]
#shift1 = erosion[1]

#roi1
sample_21 = sample2[0] + roi_centroid[0]
sample_22 = sample2[1] + roi_centroid[1]
sample_23 = sample2[2] + roi_centroid[2]
#roi2
sample_24 = sample2[0] + shift2[0]
sample_25 = sample2[1] + shift2[1]
sample_26 = sample2[2] + shift2[2]
#roi3
sample_27 = sample2[0] + shift2[3]
sample_28 = sample2[1] + shift2[4]
sample_29 =sample2[2] + shift2[5]
#roi4
sample_210= sample2[0] + shift2[10]
sample_211= sample2[1] + shift2[11]
sample_212= sample2[2] + shift2[12]
#roi5
sample_213= sample2[0] + shift2[100]
sample_214= sample2[1] + shift2[200]
sample_215= sample2[2] + shift2[300]

#adding offsets for multiple roi 
#Points from erosion 
roi_21 = ndimage.map_coordinates(seg,[sample_21.flatten(), sample_22.flatten(), sample_23.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi1[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")

roi_22 = ndimage.map_coordinates(seg,[sample_24.flatten(), sample_25.flatten(), sample_26.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi2[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi_23 = ndimage.map_coordinates(seg,[sample_27.flatten(), sample_28.flatten(), sample_29.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi3[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi_24 = ndimage.map_coordinates(seg,[sample_210.flatten(), sample_211.flatten(), sample_212.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi4[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi_25= ndimage.map_coordinates(seg,[sample_213.flatten(), sample_214.flatten(), sample_215.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi5[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 

#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone2_ROI-1_{NAME}.nrrd",roi1)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone2_ROI-2_{NAME}.nrrd",roi2)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone2_ROI-3_{NAME}.nrrd",roi3)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone2_ROI-4_{NAME}.nrrd",roi4)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone2_ROI-5_{NAME}.nrrd",roi5)

#bone3
def rotation_matrix_from_vectors(Y, Z):

    a3, b3 = (Y), (Z)
    v = np.cross(a3,b3)
    c = np.dot(a3, b3)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix3 = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix3
    #np.save("/gpfs_projects/sriharsha.marupudi/extract_rois_output/rotation_matrix-{NAME}.npy",rotation_matrix)
    #save rotation matrix 
rot3 = rotation_matrix_from_vectors(Y,Z)
np.save(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/rotation_matrix3_{NAME}.npy",rot)
rotation_result3 = np.matmul(Q,rot3)
print(rotation_result3)

points_array = np.asarray(points)
points_flatten = points_array.flatten()
a = points_flatten.shape
b =a[0]
c = b//3
points_array1 = points_flatten.reshape(3,c)
#points_array1 = points_flatten.reshape(3,8000000)
sample3 = rot3.dot(points_array1)
np.save(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/transform_coordinates3_{NAME}.npy",sample)
#save as numpy array

# Step 1: compute rotation matrix from (0,0,1) to normals[0] and normals[1]
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
# https://stackoverflow.com/questions/54445195/np-tensordot-for-rotation-of-point-clouds
# apply rotation matrix to each point: (xxt[i], yyt[i], zzt[i])

# Step 2: Shift origin from (0,0,0) to roi_centroid
#np.nonzero index results to shift 

erosion3 = np.nonzero(seg_close_erode)

s3 = erosion3[1]
shift3 = s3.flatten()

#shift0 = erosion[0]
#shift1 = erosion[1]

#roi1
sample_31 = sample3[0] + roi_centroid[0]
sample_32 = sample3[1] + roi_centroid[1]
sample_33 = sample3[2] + roi_centroid[2]
#roi2
sample_34 = sample3[0] + shift3[0]
sample_35 = sample3[1] + shift3[1]
sample_36 = sample3[2] + shift3[2]
#roi3
sample_37 = sample3[0] + shift3[3]
sample_38 = sample3[1] + shift3[4]
sample_39 =sample3[2] + shift3[5]
#roi4
sample_310= sample3[0] + shift3[10]
sample_311= sample3[1] + shift3[11]
sample_312= sample3[2] + shift3[12]
#roi5
sample_313= sample3[0] + shift3[100]
sample_314= sample3[1] + shift3[200]
sample_315= sample3[2] + shift3[300]

#adding offsets for multiple roi 
#Points from erosion 
roi_31 = ndimage.map_coordinates(seg,[sample_31.flatten(), sample_32.flatten(), sample_33.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi_31[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")

roi_32 = ndimage.map_coordinates(seg,[sample_34.flatten(), sample_35.flatten(), sample_36.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi_32[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi_33 = ndimage.map_coordinates(seg,[sample_37.flatten(), sample_38.flatten(), sample_39.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi_33[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi_34 = ndimage.map_coordinates(seg,[sample_310.flatten(), sample_311.flatten(), sample_312.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi_34[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 
roi_35= ndimage.map_coordinates(seg,[sample_313.flatten(), sample_314.flatten(), sample_315.flatten()],order=0).reshape(roi_dims)
plt.imshow(roi_35[:,:,100], interpolation="nearest", cmap="gray"); plt.axis("off")
 

#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone3_ROI-1_{NAME}.nrrd",roi1)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone3_ROI-2_{NAME}.nrrd",roi2)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone3_ROI-3_{NAME}.nrrd",roi3)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone3_ROI-4_{NAME}.nrrd",roi4)
#nrrd.write(f"/gpfs_projects/sriharsha.marupudi/extract_rois_output/Bone3_ROI-5_{NAME}.nrrd",roi5)