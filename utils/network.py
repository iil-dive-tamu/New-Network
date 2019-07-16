import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, TimeDistributed, Flatten,Dense,Dropout,MaxPooling3D,Conv3D
import tensorflow as tf
import sys
sys.path.append("..")
from keras_frcnn.RoiPoolingConv import RoiPoolingConv

class Network:
	def __init__(self, net_name, structure_model):
		"""
		Initial the chooses of networks
		:param net_name: a list from parser
		:param structure_model: a list from parser
		"""
		self.first_network = net_name[0]
		self.second_network = net_name[1]
		self.first_network_structure = structure_model[0]
		self.first_network_structure = structure_model[1]

	def get_first_network(self, input_tensor ):

		if self.first_network == 'SAME':
			if self.first_network_structure == '3D':
				return self.base(input_tensor)

	def get_second_network(self, input_tensor):
		if self.second_network == 'SAME':
			if self.first_network_structure == '3D':
				return self.base2(input_tensor)

	def base(self,input_tensor = None ):
		# Block 1
		x = Conv3D(2, 3, activation='relu', padding='same', name='block1_conv1')(input_tensor)
		x = Conv3D(2, 3, activation='relu', padding='same', name='block1_conv2')(x)

		# Block 2
		x = Conv3D(1, 3, activation='relu', padding='same', name='block2_conv1')(x)
		x = Conv3D(1, 3, activation='relu', padding='same', name='block2_conv2')(x)

		# Block 3
		x = Conv3D(1, 3, activation='relu', padding='same', name='block3_conv1')(x)
		x = Conv3D(1, 3, activation='relu', padding='same', name='block3_conv2')(x)

		# Block 4
		x = Conv3D(1, 3, activation='relu', padding='same', name='block4_conv1')(x)
		x = Conv3D(1, 3, activation='relu', padding='same', name='block4_conv2')(x)

		return x

	def base2(self,input_tensor = None ):
		# Block 1
		# output 17*17*9
		x = Conv3D(2, 3, activation='relu', padding='same', name='block1_conv1')(input_tensor)
		x = Conv3D(2, 3, activation='relu', padding='same', name='block1_conv2')(x)
		x = MaxPooling3D(pool_size=(2, 2, 2),padding='same')(x)
		# Block 2
		# output 9*9*5
		x = Conv3D(1, 3, activation='relu', padding='same', name='block2_conv1')(x)
		x = Conv3D(1, 3, activation='relu', padding='same', name='block2_conv2')(x)
		x = MaxPooling3D(pool_size=(2, 2, 2),padding='same')(x)
		# Block 3
		# output 5*5*3
		x = Conv3D(1, 3, activation='relu', padding='same', name='block3_conv1')(x)
		x = Conv3D(1, 3, activation='relu', padding='same', name='block3_conv2')(x)
		x = MaxPooling3D(pool_size=(2, 2, 2),padding='same')(x)
		# Block 4
		# output 3*3*2
		x = Conv3D(1, 3, activation='relu', padding='same', name='block4_conv1')(x)
		x = Conv3D(1, 3, activation='relu', padding='same', name='block4_conv2')(x)
		x = MaxPooling3D(pool_size=(2, 2, 2),padding='same')(x)

		# Block 5
		# output 2*2*1
		x = Conv3D(1, 3, activation='relu', padding='same', name='block5_conv1')(x)
		x = Conv3D(1, 3, activation='relu', padding='same', name='block5_conv2')(x)
		x = MaxPooling3D(pool_size=(2, 2, 2),padding='same')(x)


		x = Flatten()(x)
		x = TimeDistributed(Dense(1, activation = 'relu'))(x)

		return x
