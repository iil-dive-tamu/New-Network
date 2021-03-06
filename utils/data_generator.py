import numpy as np
import tifffile as tiff
import os
from tensorflow.keras import backend as K

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
		# if Fov.shape != Image.shape:
		# 	raise ValueError("ValueError: the shape of Fov and Image should be the same")

		shape = self.fov_size
		Fov = Fov.reshape(1, shape[0], shape[1], shape[2], 1)
		Image = Image.reshape(1, shape[0], shape[1], shape[2], 1)
		Label = Label.reshape(1, shape[0], shape[1], shape[2], 1)

		Fov_concatented = np.concatenate((Fov, Image), axis=4)

		return Fov_concatented, Label


	def data_save(self,Fov, FOV_Path, x, y, z, epoch_num,big_epoch_num):
		"""
		Saving the fov by order
		"""
		tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z, big_epoch_num,epoch_num), Fov )


	def data_remove(self, FOV_Path, epoch_num,big_epoch_num ):

		for z in range(self.start_z, self.end_z):
			for y in range(self.start_xy, self.end_xy):
				for x in range(self.start_xy, self.end_xy):

					if os.path.exists(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z,big_epoch_num, epoch_num)):
						#删除文件，可使用以下两种方法。
					    os.remove(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z, big_epoch_num,epoch_num))
					else:
						print ('no such file:%s'%FOV_Path + r'\FOV_%d%d%d_%d.tif'%(x, y, z, epoch_num))


	def data_generator(self,FOV_Path, LABEL_Path, IMAGE_Path, epoch_num,big_epoch_num):
		"""
		Generator for the first CNN
		"""
		for z in range(self.start_z, self.end_z):
			for x in range(self.start_xy, self.end_xy):
				for y in range(self.start_xy, self.end_xy):

					Fov = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,z,big_epoch_num,epoch_num) )
					Label = tiff.imread(LABEL_Path + r'\LABEL_%d%d%d.tif'%(x,y,z) )
					Image = tiff.imread(IMAGE_Path + r'\IMAGE_%d%d%d.tif'%(x,y,z) )

					yield Fov, Label, Image, x, y, z


	# def data_exchange(self,FOV_Path, epoch_num, big_epoch_num):
	# 	"""
	# 	Exchanging the values in the FOV after an epoch training of the first CNN
	# 	"""
	# 	center_x = int(self.fov_size[0]/2)
	# 	center_y = int(self.fov_size[1]/2)
	# 	center_z = int(self.fov_size[2]/2)
	#
	# # for z in range(self.start_z, self.end_z):
	# 	for x in range(self.start_xy, self.end_xy):
	# 		for y in range(self.start_xy, self.end_xy):
	# 			for z in range(self.start_z, self.end_z):
	#
	# 				Fov = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,z,big_epoch_num,epoch_num-1) )
	# 				Fov = np.squeeze(Fov)
	# 				### The first Block
	# 				if y+1 < self.end_xy:
	# 					for i in range(y+1, min(y+int(self.fov_size[0]/2)+1, self.end_xy) ):
	# 						for j in range( max(z-int(self.fov_size[2]/2), self.start_z), min(z+int(self.fov_size[2]/2)+1, self.end_z) ):
	# 							Fov_exchange = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,i,j,big_epoch_num,epoch_num-1) )
	# 							Fov_exchange = np.squeeze(Fov_exchange)
	# 							coor_difference = [0,i-y,j-z]
	# 							temp = Fov[center_x+coor_difference[0], center_y+coor_difference[1], center_z+coor_difference[2]]
	# 							Fov[center_x, center_y+coor_difference[1], center_z+coor_difference[2]] = Fov_exchange[center_x, center_y-coor_difference[1], center_z-coor_difference[2]]
	# 							Fov_exchange[center_x-coor_difference[0], center_y-coor_difference[1], center_z-coor_difference[2]] = temp
	#
	# 							tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, i, j, big_epoch_num, epoch_num-1), Fov_exchange )
	#
	#
	# 				### The second Block
	# 				if z+1 < self.end_z:
	# 					for j in range(max(z-int(self.fov_size[2]/2), self.start_z), min(z+int(self.fov_size[2]/2)+1, self.end_z)):
	# 						if j == z:
	# 							continue
	# 						Fov_exchange = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,j,big_epoch_num, epoch_num-1) )
	# 						Fov_exchange = np.squeeze(Fov_exchange)
	# 						coor_difference = j-z
	# 						temp = Fov[center_x, center_y, center_z+coor_difference]
	# 						Fov[center_x, center_y, center_z+coor_difference] = Fov_exchange[center_x, center_y, center_z-coor_difference]
	# 						Fov_exchange[center_x, center_y, center_z-coor_difference] = temp
	# 						tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, j, big_epoch_num, epoch_num-1), Fov_exchange )
	#
	# 				if x+1 < self.end_xy:
	# 					for i in range( max(y-int(self.fov_size[0]/2), self.start_xy), min(y+int(self.fov_size[0]/2)+1, self.end_xy)):
	# 						for j in range(x+1, min(x+int(self.fov_size[1]/2)+1, self.end_xy)):
	# 							for k in range(max(z-int(self.fov_size[2]/2), self.start_z), min(z+int(self.fov_size[2]/2)+1,self.end_z)):
	# 								#### problem
	#
	# 								Fov_exchange = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(j,i,k,big_epoch_num,epoch_num-1) )
	# 								Fov_exchange = np.squeeze(Fov_exchange)
	# 								coor_difference = [j-x,i-y,k-z]
	# 								temp = Fov[center_x+coor_difference[0], center_y+coor_difference[1], center_z+coor_difference[2]]
	# 								Fov[center_x+coor_difference[0], center_y+coor_difference[1], center_z+coor_difference[2]] = \
	# 									Fov_exchange[center_x-coor_difference[0], center_y-coor_difference[1], center_z-coor_difference[2]]
	#
	# 								Fov_exchange[center_x-coor_difference[0], center_y-coor_difference[1], center_z-coor_difference[2]] = temp
	#
	# 								tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(j, i, k, big_epoch_num,epoch_num-1), Fov_exchange )
	#
	#
	# 				tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z, big_epoch_num,epoch_num), Fov )

	# def data_exchange1(self,FOV_Path, epoch_num, big_epoch_num):
	# 	"""
	# 	Exchanging the values in the FOV after an epoch training of the first CNN
	# 	"""
	# 	center_x = int(self.fov_size[0]/2)
	# 	center_y = int(self.fov_size[1]/2)
	# 	center_z = int(self.fov_size[2]/2)
    #
	# # for z in range(self.start_z, self.end_z):
	# 	for x in range(self.start_xy, self.end_xy):
	# 		for y in range(self.start_xy, self.end_xy):
	# 			for z in range(self.start_z, self.end_z):
    #
	# 				Fov = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,z,big_epoch_num,epoch_num-1) )
	# 				Fov = np.squeeze(Fov)
	# 				### The first Block
	# 				for j in range( max(y-int(self.fov_size[1]/2), self.start_xy), min(y+int(self.fov_size[1]/2)+1, self.end_xy) ):
	# 					for k in range( max(z-int(self.fov_size[2]/2), self.start_z), min(z+int(self.fov_size[2]/2)+1, self.end_z) ):
	# 						for i in range( max(x-int(self.fov_size[0]/2), self.start_xy), min(x+int(self.fov_size[0]/2)+1, self.end_xy)) :
    #
	# 							if j == y and i == x and k == z:
	# 								continue
    #
	# 							Fov_exchange = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i,j,k,big_epoch_num,epoch_num-1) )
	# 							Fov_exchange = np.squeeze(Fov_exchange)
	# 							coor_difference = [i-x,j-y,k-z]
    #
	# 							Fov[center_x+coor_difference[0], center_y+coor_difference[1], center_z+coor_difference[2]] =\
	# 								Fov_exchange[center_x-coor_difference[0], center_y-coor_difference[1], center_z-coor_difference[2]]
	# 				tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z, big_epoch_num,epoch_num), Fov )
	def data_exchange(self,FOV_Path, epoch_num, big_epoch_num):
		"""
		Exchanging the values in the FOV after an epoch training of the first CNN
		"""
		center_x = int(self.fov_size[0]/2)
		center_y = int(self.fov_size[1]/2)
		center_z = int(self.fov_size[2]/2)

		mul_matrix = np.zeros((self.fov_size[0],self.fov_size[1],self.fov_size[2],self.fov_size[0]*self.fov_size[1]*self.fov_size[2])).astype('int8')
		count = -1
		for x in range(self.fov_size[0]):
			for y in range(self.fov_size[1]):
				for z in range(self.fov_size[2]):
					mul_matrix[x,y,z,count] = 1
					count -=1
		mul_matrix = K.variable(mul_matrix)

		pre_matrix = np.zeros((self.fov_size[0],self.fov_size[1],self.fov_size[2],self.fov_size[0]*self.fov_size[1]*self.fov_size[2]))
		pre_x = self.start_xy
		pre_y = self.start_xy
		pre_z = self.start_z

		#中心点位置
		for x in range(self.start_xy, self.end_xy):
			for y in range(self.start_xy, self.end_xy):
				for z in range(self.start_z, self.end_z):
					new_matrix = np.zeros(shape = pre_matrix.shape)
					Fov = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,z,big_epoch_num,epoch_num-1) )
					Fov = np.squeeze(Fov)
					### The first Block
					offset = [x-pre_x,y-pre_y,z-pre_z]
					for j in range( max(y-int(self.fov_size[1]/2), self.start_xy), min(y+int(self.fov_size[1]/2)+1, self.end_xy),2 ):
						for k in range( max(z-int(self.fov_size[2]/2), self.start_z), min(z+int(self.fov_size[2]/2)+1, self.end_z),2 ):
							for i in range( max(x-int(self.fov_size[0]/2), self.start_xy), min(x+int(self.fov_size[0]/2)+1, self.end_xy),2):
								## 读取的图片应该在fov的哪个位置
								coor = [i-x+center_x,j-y+center_y,k-z+center_z]

								if np.sum(offset)==0:
									new_matrix[coor[0],coor[1],coor[2],:] = \
										tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i,j,k,big_epoch_num,epoch_num-1)).flatten()
								elif self.overlap(coor,offset):
									new_matrix[coor[0],coor[1],coor[2],:] = \
										pre_matrix[coor[0] + offset[0],coor[1] + offset[1],coor[2] + offset[2],:]
								else:
									new_matrix[coor[0],coor[1],coor[2],:] = \
										tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i,j,k,big_epoch_num,epoch_num-1)).flatten()


					pre_matrix = K.variable(new_matrix)
					del new_matrix

					pre_x = x
					pre_y = y
					pre_z = z
					tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z, big_epoch_num,epoch_num), K.get_value( K.squeeze(K.sum(pre_matrix*mul_matrix,axis=3))))


	def data_exchange1(self,FOV_Path, epoch_num, big_epoch_num):
		"""
		Exchanging the values in the FOV after an epoch training of the first CNN
		"""
		center_x = int(self.fov_size[0]/2)
		center_y = int(self.fov_size[1]/2)
		center_z = int(self.fov_size[2]/2)

		mul_matrix = np.zeros((self.fov_size[0],self.fov_size[1],self.fov_size[2],self.fov_size[0]*self.fov_size[1]*self.fov_size[2])).astype('int8')
		count = -1
		for x in range(self.fov_size[0]):
			for y in range(self.fov_size[1]):
				for z in range(self.fov_size[2]):
					mul_matrix[x,y,z,count] = 1
					count -=1


		pre_matrix = np.zeros((self.fov_size[0],self.fov_size[1],self.fov_size[2],self.fov_size[0]*self.fov_size[1]*self.fov_size[2]))
		pre_x = self.start_xy
		pre_y = self.start_xy
		pre_z = self.start_z

		#中心点位置
		for x in range(self.start_xy, self.end_xy):
			for y in range(self.start_xy, self.end_xy):
				for z in range(self.start_z, self.end_z):
					new_matrix = np.zeros(shape = pre_matrix.shape)
					Fov = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x,y,z,big_epoch_num,epoch_num-1) )
					Fov = np.squeeze(Fov)
					### The first Block
					offset = [x-pre_x,y-pre_y,z-pre_z]
					for j in range( max(y-int(self.fov_size[1]/2), self.start_xy), min(y+int(self.fov_size[1]/2)+1, self.end_xy) ):
						for k in range( max(z-int(self.fov_size[2]/2), self.start_z), min(z+int(self.fov_size[2]/2)+1, self.end_z) ):
							for i in range( max(x-int(self.fov_size[0]/2), self.start_xy), min(x+int(self.fov_size[0]/2)+1, self.end_xy)) :
								## 读取的图片应该在fov的哪个位置
								coor = [i-x+center_x,j-y+center_y,k-z+center_z]

								if np.sum(offset)==0:
									new_matrix[coor[0],coor[1],coor[2],:] = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i,j,k,big_epoch_num,epoch_num-1)).flatten()
								elif self.overlap(coor,offset):
									new_matrix[coor[0],coor[1],coor[2],:] = pre_matrix[coor[0] + offset[0],coor[1] + offset[1],coor[2] + offset[2],:]
								else:
									new_matrix[coor[0],coor[1],coor[2],:] = tiff.imread(FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(i,j,k,big_epoch_num,epoch_num-1)).flatten()


					pre_matrix = new_matrix
					del new_matrix

					pre_x = x
					pre_y = y
					pre_z = z
					tiff.imsave( FOV_Path + r'\FOV_%d%d%d_%d_%d.tif'%(x, y, z, big_epoch_num,epoch_num), np.squeeze(np.sum(pre_matrix*mul_matrix,axis=3)))

	def overlap(self,coor,offset):
		"""
		Exchanging the values in the FOV after an epoch training of the first CNN
		"""
		a = coor[0] + offset[0]
		b = coor[1] + offset[1]
		c = coor[2] + offset[2]

		if self.fov_size[0] - a > 0 and self.fov_size[1] - b > 0 and self.fov_size[2] - c > 0 and a >= 0 and b >= 0 and c >0:
			return True
		else:
			return False


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


