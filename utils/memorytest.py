import numpy as np
import tifffile as tiff
from memory_profiler import profile

@profile
def get_data(input_image = None ,input_label = None, input_probability_map= None, Padding = 'NONE', Size = None ,Sample = False, Stride = 1, Depth_Stride = 1 ):
	"""
	This function is the preprocess of the input image,
	it will return 4 np.array: FOV, Cropped Original Image, Cropped label and Full-sized Original Image

	input_path : path of the folder of the input training image, from parser
	Padding : The model of paading used here, from parser
	Size : The size of Field Of View(FOV), from config

	Sample : Using sample mode or not, from config
	Stride : Stride on Height and Width, from config
	Depth_Stride : Stride on Depth, from config
	"""

	if Sample == True:
		raise ValueError("Sample mode has not been implemented")
	if Size == None or len(Size) != 3:
		raise ValueError("Please specify the size of FOV, it must be in the format of [x,y,z]")

	# prefilepath = r'C:\Users\sunzh\CS636\Summer project\BPN\data\train-input-split\train-input-'

	## Read the image, label and probability map
	Image = tiff.imread(input_image)
	Label = tiff.imread(input_label)
	Promap = tiff.imread(input_probability_map)
	Image = np.transpose(Image, (2,0,1)).astype('int8')
	Label = np.transpose(Label, (2,0,1)).astype('int8')
	Promap = np.transpose(Promap, (2,0,1)).astype('int8')

	print("Images are read now")

	if Image.shape != Label.shape or Image.shape != Promap.shape or Label.shape != Promap.shape:
		raise ValueError("ValueError: Image, Promap and Label should have the same size with each other")


	r_xy = int((Size[0]-1)/2)
	r_z = int((Size[2]-1) /2)

	# Start_xy = int((Size[0]-1)/2)
	# End_xy = int(Image.shape[0] - (Size[0]-1)/2)
	# Start_z = int((Size[2]-1)/2)
	# End_z = int(Image.shape[2] - (Size[2]-1)/2)
	#
	# FOV = np.zeros((1024, 1024, 100)).astype('int8')


	if Padding == 'NONE':
		Start_xy = int((Size[0]-1)/2)
		End_xy = int(Image.shape[0] - (Size[0]-1)/2)
		Start_z = int((Size[2]-1)/2)
		End_z = int(Image.shape[2] - (Size[2]-1)/2)

		FOV = np.zeros((End_xy - Start_xy +1, End_xy - Start_xy+1, End_z - Start_z +1, Size[0], Size[1], Size[2])).astype('int8')
		FOV_label = np.zeros((End_xy - Start_xy +1, End_xy - Start_xy+1, End_z - Start_z +1, Size[0], Size[1], Size[2])).astype('int8')
		FOV_Im  = np.zeros((End_xy - Start_xy +1, End_xy - Start_xy+1, End_z - Start_z +1, Size[0], Size[1], Size[2])).astype('int8')

		for i in range(Start_xy, End_xy):
			for j in range(Start_z, End_xy):
				FOV[i-Start_xy,i-Start_z,j-Start_z,:,:,:] = Promap[i-Start_xy:i-Start_xy+Size[0], i-Start_xy:i-Start_xy+Size[1], j-Start_z:j-Start_z+Size[2] ]
				FOV_label[i-Start_xy,i-Start_z,j-Start_z,:,:,:] = Label[i-Start_xy:i-Start_xy+Size[0], i-Start_xy:i-Start_xy+Size[1], j-Start_z:j-Start_z+Size[2] ]
				FOV_Im[i-Start_xy,i-Start_z,j-Start_z,:,:,:] = Image[i-Start_xy:i-Start_xy+Size[0], i-Start_xy:i-Start_xy+Size[1], j-Start_z:j-Start_z+Size[2] ]

		return FOV, FOV_label, FOV_Im, Image
	elif Padding == 'SAME':
		FOV = np.zeros((Promap.shape[0],Promap.shape[1],Promap.shape[2], Size[0], Size[1], Size[2])).astype('int8')
		FOV_label = np.zeros((Label.shape[0],Label.shape[1],Label.shape[2], Size[0], Size[1], Size[2])).astype('int8')
		FOV_Im  =  np.zeros((Image.shape[0],Image.shape[1],Image.shape[2], Size[0], Size[1], Size[2])).astype('int8')

		for i in range(0,Image.shape[0]):
			for j in range(0, Image.shape[2]):
				Start_xy = int(max(0, i- r_xy))
				End_xy = int(min(i+r_xy+1,Promap.shape[1]))
				Start_z = int(max(0,j-r_z))
				End_z = int(min(j+r_z+1,Promap.shape[2]))

				FOV[i, i, j, r_xy-(i-Start_xy):r_xy+(End_xy-i),r_xy-(i-Start_xy):r_xy+(End_xy-i), r_z-(i-Start_z):r_z+(End_z-i)] = Promap[Start_xy:End_xy, Start_xy:End_xy, Start_z:End_z]
				FOV_label[i, i, j, r_xy-(i-Start_xy):r_xy+(End_xy-i),r_xy-(i-Start_xy):r_xy+(End_xy-i), r_z-(i-Start_z):r_z+(End_z-i)] = Label[Start_xy:End_xy, Start_xy:End_xy, Start_z:End_z]
				FOV_Im[i, i, j, r_xy-(i-Start_xy):r_xy+(End_xy-i),r_xy-(i-Start_xy):r_xy+(End_xy-i), r_z-(i-Start_z):r_z+(End_z-i)] = Image[Start_xy:End_xy, Start_xy:End_xy, Start_z:End_z]

		return FOV, FOV_label, FOV_Im, Image
	else:
		raise ValueError("Padding only support SAME and NONE now")


if __name__ == '__main__':
	imagepath = r'C:\Users\sunzh\CS636\Summer project\BPN\data\train-input\train-input.tif'
	labelpath = r'C:\Users\sunzh\CS636\Summer project\BPN\data\train-labels\train-labels.tif'
	promappath = labelpath
	get_data(imagepath, labelpath, promappath,'NONE',[15,15,7], False)
