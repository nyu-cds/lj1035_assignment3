from mpi4py import MPI
import numpy

def generate_array():
	"""
	This function generates a random array of shape (10000,).
	"""
	input_arr = numpy.random.randint(10000, size=10000)
	return input_arr

def slice_array(input_arr, size):
	"""
	This function slices the input_arr to into bins.
	"""
	min_item = numpy.amin(input_arr)
	max_item = numpy.amax(input_arr)
	# Compute the length of each bin
	bin_length = (max_item - min_item) / (size - 1)
	data_to_process = []
	# Append lists of data that will be sent in all processes to 'data_to_process'
	for i in range(-1, size - 1):
		data_to_process.append([x for x in input_arr if (min_item + bin_length * i) < x <= (min_item + bin_length * (i + 1))])
	return data_to_process

def sort_data(): 
	"""
	This function sorts the data in parallel. 
	"""
	comm = MPI.COMM_WORLD
	# Get the size of the MPI.COMM_WORLD communicator
	size = comm.Get_size()
	# Get the rank of the process within the communicator
	rank = comm.Get_rank()

	if rank == 0:
		input_arr = generate_array()
		data_to_process = slice_array(input_arr, size)
	else:
		data_to_process = None

	# Send chunks of the array to different processes
	data_scattered = comm.scatter(data_to_process, root=0)

	# Sort the data of all chunks
	data_to_gather = numpy.sort(data_scattered)

	# Gather different processes to one single process
	sorted_data = comm.gather(data_to_gather, root=0)

	if rank == 0:
		# Concatenate all sorted data
		sorted_data = numpy.concatenate(sorted_data)
		print(sorted_data)

	return sorted_data

if __name__ =='__main__':
	sort_data()