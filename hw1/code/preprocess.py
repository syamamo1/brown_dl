import gzip
import numpy as np

def get_data(inputs_file_path, labels_file_path, num_examples):
	"""
	Takes in an inputs file path and labels file path, unzips both files, 
	normalizes the inputs, and returns (NumPy array of inputs, NumPy 
	array of labels). Read the data of the file into a buffer and use 
	np.frombuffer to turn the data into a NumPy array. Keep in mind that 
	each file has a header of a certain size. This method should be called
	within the main function of the model.py file to get BOTH the train and
	test data. If you change this method and/or write up separate methods for 
	both train and test data, we will deduct points.
	:param inputs_file_path: file path for inputs, something like 
	'MNIST_data/t10k-images-idx3-ubyte.gz'
	:param labels_file_path: file path for labels, something like 
	'MNIST_data/t10k-labels-idx1-ubyte.gz'
	:param num_examples: used to read from the bytestream into a buffer. Rather 
	than hardcoding a number to read from the bytestream, keep in mind that each image
	(example) is 28 * 28, with a header of a certain number.
	:return: NumPy array of inputs as float32 and labels as int8
	"""
	
	#TODO: Load inputs and labels
	#TODO: Normalize inputs

	with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
		image_header = np.frombuffer(bytestream.read(16))
		image_data = np.frombuffer(bytestream.read(), np.uint8).astype(np.float32)/255
		image_data = np.reshape(image_data, (num_examples, 784))
		
	with open(labels_file_path, 'rb') as f2, gzip.GzipFile(fileobj=f2) as bytestream2:
		label_header = np.frombuffer(bytestream2.read(8))
		label_data = np.frombuffer(bytestream2.read(), np.uint8)

	return image_data, label_data


# 60k
# a = 'train-images-idx3-ubyte.gz'
# b = 'train-labels-idx1-ubyte.gz'

# # 10k
# c = 't10k-images-idx3-ubyte.gz'

# name = a
# l_name = b

# images = "C:\\Users\smy18\workspace2\dl\hw1-mnist-syamamo1\hw1\\" + name
# labels = "C:\\Users\smy18\workspace2\dl\hw1-mnist-syamamo1\hw1\\" + l_name

# get_data(images,labels,60000)