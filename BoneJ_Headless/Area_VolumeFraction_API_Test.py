# @CommandService cs
#@ DatasetIOService io
#@ UIService ui
# @Context ctxt

#@ String NAME 
#@ String image
#@ String table_csv
#@ String outputdir

from ij import IJ
from net.imglib2.img import ImagePlusAdapter
import csv

#Prevents unicode error 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

from net.imagej import ImgPlus
from net.imagej.axis import Axes
from net.imglib2.img import Img

from net.imagej import DefaultDataset
import ij.ImagePlus;
from io.scif.img import ImgOpener
from org.bonej.utilities import SharedTable
from ij import ImagePlus
import net.imagej.ImgPlus;
from ij.io import FileSaver


#Open image from path 
input_dir = "/gpfs_projects/sriharsha.marupudi/Segmentations_Otsu_L1/"
outputdir
# outputdir = "/gpfs_projects/sriharsha.marupudi/Measurements_Test/" 
IJ.run("Clear BoneJ results");
#open input image as ImagePlus and converts to ImgPlus to be compatabile with wrapper
input_image = IJ.openImage(image)
output = ImagePlusAdapter.wrapImgPlus(input_image) 
wrapperBV = cs.run("org.bonej.wrapperPlugins.ElementFractionWrapper", False, ["inputImage",output])
wrapperBVInstance = wrapperBV.get()

#Save results to csv file 
table =SharedTable.getTable()
# print(table)
f = open(outputdir+"ROI-"+NAME+"-table.csv", 'wb')
f.write('\n')
f.write("rowname,")

for j in range(0,len(table[0])):
	f.write(table.getRowHeader(j)+",")
		
f.write('\n')

for i in range(0,len(table)):
	f.write(table.getColumnHeader(i)+",")
	for j in range(0,len(table[i])):
		f.write(str(table[i][j])+",")
	f.write('\n')
f.close()


			

			