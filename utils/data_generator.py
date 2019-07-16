import numpy as np
import tifffile as tiff
import os

class Data_Generator:
	def __init__(self, C):
		"""
		C is the config from config.py
		"""
		self.fov_size = C.field_of_view_scales
		self.cropped_size = C.cropped_size

		# we should apply SAME padding
		self.start_xy = int((self.fov_size[0]-1)/2)
		self.start_z = int((self.fov_size[2]-1)/2)
		self.end_xy = self.start_xy+self.cropped_size[0]
		self.end_z = self.start_z+self.cropped_size[2]


	def data_concatenate(self,Fov, Image, Label):
		"""
		Combination of FOV and Image matrices.
		Reshaping of FOV and label
		"""
		if Fov.shape != Image.shape:
			raise ValueError("ValueError: the shape of Fov and Image should be the same")

		shape = Fov.shape

		Fov = Fov.reshape(1, shape[0], shape[1], shape[2], 1)
		Image = Image.reshape(1, shape[0], shape[1], shape[2], 1)
		Label = Label.reshape(1, shape[0], shape[1], shape[2], 1)

		Fov_concatented = np.concatenate(Fov, Image, axis=4)

		return Fov_concatented, Label


	def data_save(self,Fov, FOV_Path, x, y, z, epoch_num,big_epoch_num):
		"""
		Saving the fov by order
		"""
		tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z, big_epoch_num,epoch_num), Fov )


	# def data_remove(self, FOV_Path, x, y, z, epoch_num):
	# 	if os.path.exists(FOV_Path + r'\FOV_%d%d%d_%d.tif'%(x, y, z, epoch_num)):
	# 		#删除文件，可使用以下两种方法。
	# 	    os.remove(FOV_Path + r'\FOV_%d%d%d_%d.tif'%(x, y, z, epoch_num))
	# 	else:
	# 		print ('no such file:%s'%FOV_Path + r'\FOV_%d%d%d_%d.tif'%(x, y, z, epoch_num))


	def data_generator(self,FOV_Path, LABEL_Path, IMAGE_Path, epoch_num,big_epoch_num):
		"""
		Generator for the first CNN
		"""
		for x in range(self.start_xy, self.end_xy):
			for y in range(self.start_xy, self.end_xy):
				for z in range(self.start_z, self.end_z):
					Fov = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,z,big_epoch_num,epoch_num) )
					Label = tiff.imread(LABEL_Path + r'\LABEL_%d%d%d.tif'%(x,y,z) )
					Image = tiff.imread(IMAGE_Path + r'\IMAGE_%d%d%d.tif'%(x,y,z) )

					yield Fov, Label, Image, x, y, z


	def data_exchange(self,FOV_Path, epoch_num, big_epoch_num):
		"""
		Exchanging the values in the FOV after an epoch training of the first CNN
		"""
		center_x = int(self.fov_size[0]/2)
		center_y = int(self.fov_size[1]/2)
		center_z = int(self.fov_size[2]/2)

		for z in range(self.start_z, self.end_z):

			for y in range(self.start_xy, self.end_xy):

				for x in range(self.start_xy, self.end_xy):

					Fov = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,z,big_epoch_num,epoch_num-1) )

					### The first Block
					if x+1 < self.end_xy:
						for i in range(x+1, min(x+int(self.fov_size[0]/2)+1, self.end_xy) ):
							for j in range(z, min(z+int(self.fov_size[2]/2)+1, self.end_z) ):
								Fov_exchange = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i,y,j,big_epoch_num,epoch_num-1) )
								coor_difference = [i-x,0,j-z]
								temp = Fov[center_x+coor_difference[0], center_y, center_z+coor_difference[2]]
								Fov[center_x+coor_difference[0], center_y, center_z+coor_difference[2]] = Fov_exchange[center_x-coor_difference[0], center_y, center_z-coor_difference[2]]
								Fov_exchange[center_x-coor_difference[0], center_y, center_z-coor_difference[2]] = temp
								tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i, y, j, big_epoch_num, epoch_num-1), Fov_exchange )

					### The second Block
					if z+1 < self.end_z:
						for j in range(z+1, min(z+int(self.fov_size[2]/2)+1, self.end_z)):

							Fov_exchange = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,j,big_epoch_num, epoch_num-1) )
							coor_difference = j-z
							temp = Fov[center_x, center_y, center_z+coor_difference]
							Fov[center_x, center_y, center_z+coor_difference] = Fov_exchange[center_x, center_y, center_z-coor_difference]
							Fov_exchange[center_x, center_y, center_z-coor_difference] = temp
							tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, j, big_epoch_num, epoch_num-1), Fov_exchange )

					if y+1 < self.end_xy:
						for i in range(  max(x-int(self.fov_size[0]/2), self.start_xy), min(x+int(self.fov_size[0]/2)+1, self.end_xy)):
							for j in range(y+1, min(y+int(self.fov_size[1]/2)+1, self.end_xy)):
								for k in range(z, min(z+int(self.fov_size[2]/2)+1,self.end_z)):
									#### problem
									Fov_exchange = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i,y,j,big_epoch_num,epoch_num-1) )
									coor_difference = [i-x,j-y,k-z]
									temp = Fov[center_x+coor_difference[0], center_y+coor_difference[1], center_z+coor_difference[2]]
									Fov[center_x+coor_difference[0], center_y+coor_difference[1], center_z+coor_difference[2]] = \
										Fov_exchange[center_x-coor_difference[0], center_y-coor_difference[1], center_z-coor_difference[2]]

									Fov_exchange[center_x-coor_difference[0], center_y, center_z-coor_difference[2]] = temp
									tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i, y, j, big_epoch_num,epoch_num-1), Fov_exchange )

					tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z, big_epoch_num,epoch_num), Fov )


	def stack_generator(self, FOV_Path, IMAGE_Path, final_epoch_num,big_epoch_num):
		"""
		Generator of the second CNN
		"""

		for x in range(self.start_xy, self.end_xy):
			for y in range(self.start_xy, self.end_xy):
				for z in range(self.start_z, self.end_z):
					Fov = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,z,big_epoch_num,final_epoch_num) )
					Image = tiff.imread(IMAGE_Path + r'\IMAGE_%d%d%d.tif'%(x,y,z) )

					shape = Fov.shape
					Fov = Fov.reshape(1, shape[0], shape[1], shape[2], 1)
					Image = Image.reshape(1, shape[0], shape[1], shape[2], 1)
					Fov_concatented = np.concatenate(Fov, Image, axis=4)

					yield Fov_concatented, x-self.start_xy, y-self.start_xy, z-self.start_z


