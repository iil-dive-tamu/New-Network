class Config:

	def __init__(self):

		# The network used in Training
		self.network = ''

		# setting for data augmentation
		self.use_horizontal_flips = False
		self.use_vertical_flips = False
		self.rot_90 = False

		# field of view scales
		self.field_of_view_scales = [33, 33, 17]

		# Stop threshold for the frist and second CNN
		self.first_threshold = 0.1
		self.second_threshold = 0.1

		# Enable or Disable classes balance
		self.balanced_classes = False

		# Boundary Pixel/Voxel setting
		self.boundary_padding = 'NONE'

		# Enable sample FOV or disable. sample_stride works in Height and Width, sample_rate works in depth or channel
		self.sample_fov = False
		self.stride_hw = 1
		self.stride_depth = 1

		# The real input size of FOV
		self.cropped_size = [128,128,25]
		self.base_net_weights = None

		# Flag of mean=0 std=1 transformation
		self.featurewise_center = True
		self.featurewise_std_normalization = True

		# Threshold to stop the iteration
		self.first_threshold = 0.001
		self.second_threshold = 0.001
