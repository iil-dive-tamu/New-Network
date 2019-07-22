import sys
sys.path.append("..")

import data_generator
import os.path
import numpy as np
import tifffile as tiff
import unittest
import itertools
from preprocessing import Data
from config import Config



C = Config()
C.field_of_view_scales = [3, 3, 1]
C.cropped_size = [5, 5, 1]
D = data_generator.Data_Generator(C)




class Test(unittest.TestCase):
	def test_data_generator(self):
		FOV_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\a.tif'
		LABEL_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\a.tif'
		IMAGE_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\a.tif'

		data_creator = Data(FOV_Path, LABEL_Path, IMAGE_Path,C, False)
		Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data'
		data_creator.create_data(Path)

		first_data_generator = itertools.cycle(D.data_generator(Path+r'\FOV', Path+r'\LABEL', Path+r'\IMAGE',0, 0))

		for z in range(D.start_z, D.end_z):
			for x in range(D.start_xy, D.end_xy):
				for y in range(D.start_xy, D.end_xy):


					Fov, Label, Image, i, j, k = next(first_data_generator)

					self.assertEqual(x, i)
					self.assertEqual(y, j)
					self.assertEqual(z, k)

					self.assertEqual( np.sum(Fov-tiff.imread(r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\FOV\FOV_%d%d%d_%d_%d.tif'%(x,y,z,0,0))),0)
					self.assertEqual( np.sum(Label- tiff.imread(r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\LABEL\LABEL_%d%d%d.tif'%(x,y,z))),0)
					self.assertEqual( np.sum(Image- tiff.imread(r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\IMAGE\IMAGE_%d%d%d.tif'%(x,y,z))),0)


	def test_data_exchange(self):
		FOV_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\a.tif'
		LABEL_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\a.tif'
		IMAGE_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\a.tif'

		data_creator = Data(FOV_Path, LABEL_Path, IMAGE_Path,C, False)
		Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data'
		data_creator.create_data(Path)

		D.data_exchange(Path+r'\FOV', 1, 0 )
		first_data_generator = itertools.cycle(D.data_generator(Path+r'\FOV', Path+r'\LABEL', Path+r'\IMAGE',1, 0))


		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 4)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 12)


		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 24)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 20)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 36)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 63)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 72)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 81)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 60)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 66)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 108)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 117)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 126)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 90)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 96)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 153)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 162)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 171)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 120)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 84)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 22*6)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 23*6)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 24*6)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 100)


	def test_data_exchange_3d(self):
		C1 = Config()
		C1.field_of_view_scales = [3, 3, 3]
		C1.cropped_size = [5, 5, 3]
		D1 = data_generator.Data_Generator(C1)

		FOV_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\b.tif'
		LABEL_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\b.tif'
		IMAGE_Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data\b.tif'

		data_creator = Data(FOV_Path, LABEL_Path, IMAGE_Path,C1,False)
		Path = r'C:\Users\sunzh\CS636\Summer project\BPN\utils\test\test_data'
		data_creator.create_data(Path)

		print(data_creator.promap)

		D1.data_exchange(Path+r'\FOV', 1, 0 )
		first_data_generator = itertools.cycle(D1.data_generator(Path+r'\FOV', Path+r'\LABEL', Path+r'\IMAGE',1, 0))


		Fov, Label, Image, i, j, k = next(first_data_generator)
		print(Fov)
		self.assertEqual(np.sum(Fov), 8)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 24)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 36)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 48)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 40)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 72)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 7*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 8*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 9*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 120)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 11*12)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 12*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 13*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 14*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 15*12)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 16*12)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 17*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 18*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 19*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 20*12)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 21*8)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 22*12)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 23*12)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 24*12)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 25*8)

		#################### Ch2

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 1*12)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 2*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 3*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 4*18)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 5*12)
		#
		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 6*18)
		#
		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 7*27)
		#
		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 8*27)
		#
		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 9*27)
		#
		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 10*6*3)

		Fov, Label, Image, i, j, k = next(first_data_generator)
		self.assertEqual(np.sum(Fov), 11*6*3)


if __name__ == '__main__':
	unittest.main()
