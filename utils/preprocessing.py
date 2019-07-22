import numpy as np
import tifffile as tiff
from memory_profiler import profile


class Data:
	def __init__(self, input_image = None ,input_label = None, input_probability_map= None, Config = None, LLR = True):
		"""
		:param input_image: Folder path of input image, coming from parser
		:param input_label:  Folder path of input label, coming from parser
		:param input_probability_map: Folder path of input probability, coming from parser, generated by another algorithm
		:param Config: class from config.py
		"""
		if Config == None:
			raise ValueError('Config must be specified')

		self.padding = Config.boundary_padding
		self.size = Config.field_of_view_scales
		self.stride_hw = Config.stride_hw
		self.stride_depth = Config.stride_depth
		self.cropped_size = Config.cropped_size

		## Function pf sampling is not finished
		self.sample = Config.sample_fov
		if self.sample == True:
			raise ValueError("ValueError: Sample mode has not been implemented")

		## Input Exam
		if self.size == None or len(self.size) != 3:
			raise ValueError("ValueError: Please specify the size of FOV, it must be in the format of [x,y,z]")
		if self.cropped_size[0] < self.size[0] or self.cropped_size[1] < self.size[1] or self.cropped_size[2] < self.size[2]:
			print('The size of cropped image is %d, %d, %d'%(self.cropped_size[0],self.cropped_size[1],self.cropped_size[2]))
			print('The size of fov is %d, %d, %d'%(self.size[0],self.size[1],self.size[2]))
			raise ValueError('ValueError: The size of cropped image must larger then that of the Field of View')

		## Read the image, label and probability map
		Image = tiff.imread(input_image)
		Label = tiff.imread(input_label)
		Promap = tiff.imread(input_probability_map)
		Image = np.transpose(Image, (1,2,0)).astype('float32')
		Label = np.transpose(Label, (1,2,0)).astype('float32')
		Promap = np.transpose(Promap, (1,2,0)).astype('float32')

		# Image Shape Exam
		if Image.shape != Label.shape or Image.shape != Promap.shape or Label.shape != Promap.shape:
			raise ValueError("ValueError: Image, Promap and Label should have the same size with each other")
		if Image.shape[0] < self.cropped_size[0] or Image.shape[1] < self.cropped_size[1] or Image.shape[2] < self.cropped_size[2]:
			raise ValueError("ValueError: The size of cropped image should smaller then that of the original image",
			                 'The size of cropped image is %d, %d, %d'%(self.cropped_size[0],self.cropped_size[1],self.cropped_size[2]),
			                 'The size of Image is %d, %d, %d'%(Image.shape[0],Image.shape[1],Image.shape[2]))

		# Crop the image according to the config
		self.image = Image[0:self.cropped_size[0], 0:self.cropped_size[1], 0:self.cropped_size[2]]
		self.label = Label[0:self.cropped_size[0], 0:self.cropped_size[1], 0:self.cropped_size[2]]
		self.promap = Promap[0:self.cropped_size[0], 0:self.cropped_size[1], 0:self.cropped_size[2]]

		# Get 0-1 label and apply LLR transormation
		self.label = self.label_making(self.label)

		if LLR:
			self.promap = self.LLR(self.promap)

		# Mean = 0, Std = 1, only for input image
		if Config.featurewise_center == True:
			self.image = self.featurewise_center(self.image)
		if Config.featurewise_std_normalization == True:
			self.image = self.featurewise_std_normalization(self.image)

		print("Images are read now")


	def featurewise_center(self, image):
		"""
		:param image: only input image
		:return: input image with mean = 0
		"""
		mean = np.mean(image).astype('float32')
		return image - mean

	def featurewise_std_normalization(self, image):
		"""
		:param image: only input image
		:return: input image with std = 1
		"""
		std = np.std(image).astype('float32')
		return image/std

	def label_making(self, image):
		"""
		The original label has different num for different patch of cell, but we want 0-1 label
		:param image: only input label
		:return: 0-1 label
		"""
		return np.where(image>0, 1, 0)


	def create_data(self, path = None):
		"""
		This function is the preprocessing of the first CNN. Each fov and corresponding image, labels will be cropped and
		save to HDD, SAME padding is applied here.

		:param path: path to save the matrices, should have 3 sub-folders with name of FOV, LABEL, IMAGE
		:return: The path of FOV, IMAGE, LABEL with the size of them, the number of FOV
		"""
		if path == None:
			raise ValueError('path of data folder must be specified')
		elif type(path) != str:
			raise ValueError('path should be string')

		r_xy = int((self.size[0]-1)/2)
		# r_z = max( int((self.size[2]-1) /2), 1)
		r_z = max( int((self.size[2]-1) /2), 0)

		FOV_num = 0

		FOV_same = np.zeros((self.promap.shape[0]+self.size[0]-1, self.promap.shape[1]+self.size[1]-1, self.promap.shape[2]+self.size[2]-1)).astype('float32')
		FOV_Label_same = np.zeros((self.label.shape[0]+self.size[0]-1, self.label.shape[1]+self.size[1]-1, self.label.shape[2]+self.size[2]-1)).astype('float32')
		FOV_Im_same  =  np.zeros((self.image.shape[0]+self.size[0]-1, self.image.shape[1]+self.size[1]-1, self.image.shape[2]+self.size[2]-1)).astype('float32')

		FOV_same[r_xy:r_xy+self.cropped_size[0], r_xy:r_xy+self.cropped_size[1], r_z:r_z+self.cropped_size[2] ] = self.promap
		FOV_Label_same[r_xy:r_xy+self.cropped_size[0], r_xy:r_xy+self.cropped_size[1], r_z:r_z+self.cropped_size[2] ] = self.label
		FOV_Im_same[r_xy:r_xy+self.cropped_size[0], r_xy:r_xy+self.cropped_size[1], r_z:r_z+self.cropped_size[2] ] = self.image

		for x in range(r_xy, r_xy+self.cropped_size[0]):
			for y in range(r_xy, r_xy+self.cropped_size[1]):
				for z in range(r_z, r_z+self.cropped_size[2]):

					FOV = FOV_same[x-r_xy:x+r_xy+1, y-r_xy:y+r_xy+1, z-r_z:z+r_z+1]
					FOV_label = FOV_Label_same[x-r_xy:x+r_xy+1, y-r_xy:y+r_xy+1, z-r_z:z+r_z+1]
					FOV_Im = FOV_Im_same[x-r_xy:x+r_xy+1, y-r_xy:y+r_xy+1, z-r_z:z+r_z+1]

					FOV_num += 1

					tiff.imsave( path + r'\FOV\FOV_%d%d%d_0_0.tif'%(x,y,z), FOV )
					tiff.imsave( path + r'\LABEL\LABEL_%d%d%d.tif'%(x,y,z), FOV_label )
					tiff.imsave( path + r'\IMAGE\IMAGE_%d%d%d.tif'%(x,y,z), FOV_Im )

		shape = [FOV.shape[0], FOV.shape[1], FOV.shape[2], 2]

		return path + r'\FOV' , path + r'\LABEL', path + r'\IMAGE' , shape, FOV_num



	def for_next_iteration(self, FOV_Path, output_probability_map, big_epoch_num):
		"""
		This function is the preprocessing of the output of second CNN to make data for the first CNN in the next round.
		Combing each output probability map by the second CNN and the original image.
		Saving the combined matrices to HDD.

		:param FOV_Path: Path to read the FOV
		:param output_probability_map: Output of the 2nd CNN
		:param big_epoch_num: The number of the big iteration loop
		:return: The size of each FOV the number of FOVs for testing
		"""
		FOV_num = 0

		r_xy = int((self.size[0]-1)/2)
		r_z = max( int((self.size[2]-1) /2), 0)
		output_probability_map = self.LLR(output_probability_map)

		FOV_same = np.zeros((self.promap.shape[0]+self.size[0]-1, self.promap.shape[1]+self.size[1]-1, self.promap.shape[2]+self.size[2]-1)).astype('float32')
		FOV_same[r_xy:r_xy+self.cropped_size[0], r_xy:r_xy+self.cropped_size[1], r_z:r_z+self.cropped_size[2] ] = output_probability_map

		for x in range(r_xy, r_xy+self.cropped_size[0]):
			for y in range(r_xy, r_xy+self.cropped_size[1]):
				for z in range(r_z, r_z+self.cropped_size[2]):

					FOV = FOV_same[x-r_xy:x+r_xy+1, y-r_xy:y+r_xy+1, z-r_z:z+r_z+1]
					FOV_num += 1

					tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_0.tif'%(x,y,z,big_epoch_num), FOV )

		shape = [FOV.shape[0], FOV.shape[1], FOV.shape[2], 2]

		return  shape, FOV_num


	def LLR(self, Promap):
		"""
		Apply Log-LogistRatio transformation
		:param Label: The input lable
		:return: The input lable with LLR value
		"""
		L = Promap.astype('float64')
		epsilon = 0.00001
		Promap = np.log2(L/((1-L)+epsilon))
		return Promap


# def create_data(self, path = None):
# 		"""
# 		This function is the preprocess of the input image. There will be 2 stacks saved to the HHD
# 		The first stack is composed by the matrix from the original image and probability
#
# 		input_path : path of the folder of the input training image, from parser
# 		Padding : The model of padding used here, from parser
# 		Size : The size of Field Of View(FOV), from config
#
# 		Sample : Using sample mode or not, from config
# 		Stride : Stride on Height and Width, from config
# 		Depth_Stride : Stride on Depth, from config
# 		"""
# 		if path == None:
# 			raise ValueError('path of data folder must be specified')
# 		elif type(path) != str:
# 			raise ValueError('path should be string')
#
# 		r_xy = int((self.size[0]-1)/2)
# 		r_z = max( int((self.size[2]-1) /2), 1)
#
# 		FOV_num = 0
# 		if self.padding == 'NONE':
# 			Start_xy = int((self.size[0]-1)/2)
# 			End_xy = int(self.image.shape[0] - (self.size[0]-1)/2)
#
# 			Start_z = int((self.size[2]-1)/2)
# 			End_z = int(self.image.shape[2] - (self.size[2]-1)/2)
#
# 			for x in range(Start_xy, End_xy):
# 				for y in range(Start_xy, End_xy):
# 					for z in range(Start_z, End_z):
# 						FOV = self.promap[x-Start_xy:x-Start_xy+self.size[0], y-Start_xy:y-Start_xy+self.size[1], z-Start_z:z-Start_z+self.size[2] ]
# 						FOV_label = self.label[x-Start_xy:x-Start_xy+self.size[0], y-Start_xy:y-Start_xy+self.size[1], z-Start_z:z-Start_z+self.size[2] ]
# 						FOV_Im = self.image[x-Start_xy:x-Start_xy+self.size[0], y-Start_xy:y-Start_xy+self.size[1], z-Start_z:z-Start_z+self.size[2] ]
#
# 						FOV_num += 1
#
# 						tiff.imsave( path + r'\FOV\FOV_%d%d%d_0_0.tif'%(x,y,z), FOV )
# 						tiff.imsave( path + r'\LABEL\LABEL_%d%d%d.tif'%(x,y,z), FOV_label )
# 						tiff.imsave( path + r'\IMAGE\IMAGE_%d%d%d.tif'%(x,y,z), FOV_Im )
#
# 			FOV = FOV.reshape((FOV.shape[0], FOV.shape[1], FOV.shape[2], 2))
#
# 			return path + r'\FOV' , path + r'\LABEL', path + r'\IMAGE' , FOV.shape, FOV_num
#
#
# 		elif self.padding == 'SAME':
#
# 			FOV_same = np.zeros((self.promap.shape[0]+self.size[0]-1, self.promap.shape[1]+self.size[1]-1, self.promap.shape[2]+self.size[2]-1)).astype('float32')
# 			FOV_Label_same = np.zeros((self.label.shape[0]+self.size[0]-1, self.label.shape[1]+self.size[1]-1, self.label.shape[2]+self.size[2]-1)).astype('float32')
# 			FOV_Im_same  =  np.zeros((self.image.shape[0]+self.size[0]-1, self.image.shape[1]+self.size[1]-1, self.image.shape[2]+self.size[2]-1)).astype('float32')
#
# 			FOV_same[r_xy:r_xy+self.cropped_size[0], r_xy:r_xy+self.cropped_size[1], r_z:r_z+self.cropped_size[2] ] = self.promap
# 			FOV_Label_same[r_xy:r_xy+self.cropped_size[0], r_xy:r_xy+self.cropped_size[1], r_z:r_z+self.cropped_size[2] ] = self.label
# 			FOV_Im_same[r_xy:r_xy+self.cropped_size[0], r_xy:r_xy+self.cropped_size[1], r_z:r_z+self.cropped_size[2] ] = self.image
#
# 			for x in range(r_xy, r_xy+self.cropped_size[0]):
# 				for y in range(r_xy, r_xy+self.cropped_size[1]):
# 					for z in range(r_z, r_z+self.cropped_size[2]):
#
# 						FOV = FOV_same[x-r_xy:x+r_xy+1, y-r_xy:y+r_xy+1, z-r_z:z+r_z+1]
# 						FOV_label = FOV_Label_same[x-r_xy:x+r_xy+1, y-r_xy:y+r_xy+1, z-r_z:z+r_z+1]
# 						FOV_Im = FOV_Im_same[x-r_xy:x+r_xy+1, y-r_xy:y+r_xy+1, z-r_z:z+r_z+1]
#
# 						FOV_num += 1
#
# 						tiff.imsave( path + r'\FOV\FOV_%d%d%d_0_0.tif'%(x,y,z), FOV )
# 						tiff.imsave( path + r'\LABEL\LABEL_%d%d%d.tif'%(x,y,z), FOV_label )
# 						tiff.imsave( path + r'\IMAGE\IMAGE_%d%d%d.tif'%(x,y,z), FOV_Im )
#
# 			shape = [FOV.shape[0], FOV.shape[1], FOV.shape[2], 2]
#
# 			return path + r'\FOV' , path + r'\LABEL', path + r'\IMAGE' , shape, FOV_num
# 		else:
# 			raise ValueError("Padding only support SAME and NONE now")



if __name__ == '__main__':
	imagepath = r'C:\Users\sunzh\CS636\Summer project\BPN\data\train-input\train-input.tif'
	labelpath = r'C:\Users\sunzh\CS636\Summer project\BPN\data\train-labels\train-labels.tif'
	promappath = labelpath
