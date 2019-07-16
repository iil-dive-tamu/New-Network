import numpy as np
from optparse import OptionParser
from utils import config,preprocessing,losses
from utils.network import Network
from utils.data_generator import Data_Generator
import sys
import pickle
import itertools


from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Progbar

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--image", dest="train_image", help="Path to training image.")
parser.add_option("--promap", dest="train_probability_map", help="Path to training probability map.")
parser.add_option("--label", dest="train_label", help="Path to training label.")
parser.add_option("--cropped_size", dest='cropped_size',help='The size of image that network will use in the end',default=[128,128,25])

parser.add_option("--network", dest="network", help="Base network to use. Supports ." , default = ['SAME','SAME'])
parser.add_option("--structure", dest="structure", help="Base structure of network. Supports 3D or TD2D." , default = ['3D','3D'])

parser.add_option("--num_first_epochs", dest="num_first_epochs", help="The number of epoch of the first CNN.")
parser.add_option("--num_epochs", dest="num_epochs", help="The number of epoch of the whole network.")

parser.add_option("--config_filename", dest="config_filename", help=
				"Location to store all the metadata related to the training (to be used when testing).",
				default="config.pickle")

parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.", default=['./weight1.hdf5','./weight2.hdf5'])
parser.add_option("--input_weight_path", dest="input_weight_path", help="Input path for weights. If not specified, will initial the network by keras.", default = False)
parser.add_option("--data_augment", type="bool", dest="data_augment", help="True to enable, False to disable.", default=False)

(options, args) = parser.parse_args()

if not options.train_image:   # if filename is not given
	parser.error('Error: path to training data must be specified. Pass --image to command line')
if not options.train_probability_map:   # if filename is not given
	parser.error('Error: path to training probability map must be specified. Pass --promap to command line')
if not options.train_label:   # if filename is not given
	parser.error('Error: path to training label must be specified. Pass --label to command line')
if not options.network:
	parser.error('Error: network to use for training must be specified. Pass --network to comman line')

if options.cropped_size == None or len(options.cropped_size) != 3:
	parser.error('Error: size of cropped image must be specified')
elif type(options.cropped_size[0]) != int or type(options.cropped_size[1])!= int or type(options.cropped_size[0]) != int:
	parser.error('Error: size of cropped image must be list of int')


# pass the settings from the command line, and persist them in the config object
C = config.Config()


C.model_path = options.output_weight_path
C.network = options.network
C.cropped_size = options.cropped_size

# check if weight path was passed via command line
if options.input_weight_path:
	C.base_net_weights = options.input_weight_path

config_output_filename = options.config_filename

# Save the config for evaluation
with open(config_output_filename, 'wb') as config_f:
	pickle.dump(C,config_f)
	print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(config_output_filename))
########################################################################
# Acquisition of Data: FOV, Cropped Original Image and Whole Image
Preprocessing = preprocessing.Data(options.train_image, options.train_label, options.train_probability_map, options.padding, C)
FOV_Path, LABEL_Path, IMAGE_Path, FOV_size, Fov_num = Preprocessing.create_data(r'\data')

# Input of 2 models
FOV_input = Input(shape = FOV_size)
Stack_input = Input(shape = FOV_size)

# Network of two CNNs
Net = Network(options.network, options.structure)
First_Network_Output = Net.get_first_network(FOV_input)
Second_Network_Output = Net.get_second_network(Stack_input)

# Build the model
First_Model = Model(FOV_input, First_Network_Output)
Second_Model = Model(Stack_input, Second_Network_Output)

# load the weight
if options.input_weight_path:
	First_Model.load_weights(options.input_weight_path[0], by_name=True)
	Second_Model.load_weights(options.input_weight_path[1], by_name = True)

# Complie the model
optimizer = RMSprop(lr=1e-5)
First_Model.comile(optimizer = optimizer, loss = losses.binary_crossentropy )
Second_Model.comile(optimizer = optimizer, loss = losses.binary_crossentropy)

# Threshold and epochs for 2 CNNs
first_threshold = C.first_threshold
second_threshold = C.second_threshold
num_first_epochs = int(options.num_first_epochs)
num_epochs = int(options.num_epochs)


# Initial Data class
Data = Data_Generator(C)

# The start of
for big_epoch_num in range(num_epochs):
	###########################################################################
							#First CNN
	###########################################################################
	first_loss = np.zeros(num_first_epochs)
	final_epoch_num = 0
	for epoch_num in range(num_first_epochs):
		progbar = Progbar(num_first_epochs)
		print('Epoch of the First CNN {}/{}'.format(epoch_num+1, num_first_epochs))
		loss = 0
		count = 0
		change_matrix = 0

		first_data_generator = itertools.cycle(Data.data_generator(FOV_Path, LABEL_Path, IMAGE_Path,epoch_num, big_epoch_num))
		while count <= Fov_num:
			try:
				## 设计思想，读取一次 训练一个batch，使用itertools， 一旦停止即进行下一个 for循环

				Fov, Label, Image, x, y, z = next(first_data_generator)
				Fov, Label = Data.data_concatenate(Fov, Label, Image)
				count += 1
				current_loss = First_Model.train_on_batch(Fov, Label)
				loss += current_loss
				Fov_predict = First_Model.predict_on_batch(Fov)

				Data.data_save(Fov_predict, FOV_Path, x, y, z, epoch_num,big_epoch_num)

				changes = Fov_predict - Fov
				change_matrix += changes
				progbar.update(epoch_num+1, [('loss',current_loss),('changes',changes)])

				######退出机制######
			except Exception as e:
				print('Exception: {}'.format(e))
				continue

		Data.data_exchange(FOV_Path, epoch_num+1, big_epoch_num )

		final_epoch_num = epoch_num+1
		first_loss[epoch_num] = loss/count
		change = np.sum(change_matrix)/count

		print('loss of the %d round in first CNN is %f'%(epoch_num , loss/count))
		if loss < first_threshold or change < first_threshold:
			break
		First_Model.save(options.output_weight_path[0])
	print('Training of the first CNN is finished and loss is %F'%(np.mean(first_loss)))

	###########################################################################
							#Second CNN
	###########################################################################

	print("Training of the second network starts")
	loss = 0
	count = 0
	Label = Preprocessing.label
	second_data_generator = itertools.cycle(Data.stack_generator(FOV_Path, IMAGE_Path,final_epoch_num, big_epoch_num))
	output_probability_map = np.zeros(shape=Data.cropped_size)

	while count <= Fov_num:
		try:
			## 设计思想，读取一次 训练一个batch，使用itertools， 一旦停止即进行下一个 for循环
			Fov, x, y, z = next(second_data_generator)
			count += 1
			loss += Second_Model.train_on_batch(Fov, Label[x,y,z])
			Fov_predict = Second_Model.predict_on_batch(Fov)
			output_probability_map[x,y,z] = Fov_predict
		except Exception as e:
			print('Exception: {}'.format(e))
			continue

	Second_Model.save(options.output_weight_path[1])
	print('loss of the second CNN in %d round is %f'%(big_epoch_num, loss/count))
	if loss/count < second_threshold:
		print('Probability map satisfy the requirement and training finished in %d th round'%(big_epoch_num))
		break
	else:
		test_shape, test_num = Preprocessing.for_next_iteration(FOV_Path, output_probability_map, big_epoch_num)

		if test_shape != FOV_size:
			print("The size of generated data %d and the original data %d are not the same."%(test_shape, FOV_size))
		elif test_num !=Fov_num:
			print("The num of generated data %d and the original data %d are not the same."%(test_num, Fov_num))
		else:
			print("We should generate Fov for the next iteration")


































