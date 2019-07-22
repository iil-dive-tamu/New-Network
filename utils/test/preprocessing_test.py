import numpy as np
import tifffile as tiff
import unittest
import sys
sys.path.append("..")

from preprocessing import Data
from config import Config
import os.path

C = Config()
C.field_of_view_scales = [5, 5, 3]
C.cropped_size = [5, 5, 3]

imagepath = r'C:\Users\sunzh\CS636\Summer project\BPN\data\train-input\train-input.tif'
labelpath = r'C:\Users\sunzh\CS636\Summer project\BPN\data\train-labels\train-labels.tif'
promappath = r'C:\Users\sunzh\CS636\Summer project\BPN\data\PROMAP\train-promap.tif'

Image = tiff.imread(imagepath)
Label = tiff.imread(labelpath)
Promap = tiff.imread(promappath)

Image = np.transpose(Image, (1,2,0)).astype('float32')
Label = np.transpose(Label, (1,2,0)).astype('float32')
Promap = np.transpose(Promap, (1,2,0)).astype('float32')

class TestPreprocessing(unittest.TestCase):

	def test_init(self):
		D = Data(imagepath, labelpath, promappath,C)
		self.assertEqual(D.padding, C.boundary_padding)
		self.assertEqual(D.size, C.field_of_view_scales)
		self.assertEqual(D.sample, C.sample_fov)
		self.assertEqual(D.stride_hw, C.stride_hw)
		self.assertEqual(D.stride_depth, C.stride_depth)
		self.assertEqual(D.cropped_size, C.cropped_size)

		self.assertAlmostEqual( int(np.sum( D.image -  D.featurewise_std_normalization( D.featurewise_center( Image[0:C.cropped_size[0], 0:C.cropped_size[1], 0:C.cropped_size[2]] ) ) ) ), 0)
		self.assertEqual( np.sum( D.label -  D.label_making(  Label[0:C.cropped_size[0], 0:C.cropped_size[1], 0:C.cropped_size[2]] ) ), 0)
		self.assertEqual( np.sum( D.promap - D.LLR(Promap[0:C.cropped_size[0], 0:C.cropped_size[1], 0:C.cropped_size[2]] )), 0)

		print('1st test case finished')

	def test_createdata_SAME(self):
		D = Data(imagepath, labelpath, promappath,C)
		Path = r'C:\Users\sunzh\CS636\Summer project\BPN\data'
		D.create_data(Path)

		r_xy = int((D.size[0]-1)/2)
		r_z = max( int((D.size[2]-1) /2), 1)

		for x in range(r_xy, r_xy+D.cropped_size[0]):
			for y in range(r_xy, r_xy+D.cropped_size[1]):
				for z in range(r_z, r_z+D.cropped_size[2]):
					self.assertTrue(os.path.isfile(Path + r'\FOV\FOV_%d%d%d_0_0.tif'%( x,y,z)))
					self.assertTrue(os.path.isfile(Path + r'\LABEL\LABEL_%d%d%d.tif'%( x,y,z)))
		print('2nd test case finished')


if __name__ == '__main__':
	unittest.main()
