from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
# Get the size of the MPI.COMM_WORLD communicator
size = comm.Get_size()
# Get the rank of the process within the communicator
rank = comm.Get_rank()

# Create a buffer
value = numpy.ones(1, dtype=int)

if rank == 0:
	while True:
		num = input('Please enter an integer less than 100: ')
		# Handle exception when the input is not an integer
		try:
			value[0] = int(num)
		except ValueError:
			print('This is not a valid integer.')
			continue
		# Check whether or not it is less than 100 when the input is an integer
		if int(num) < 100:
			break
		else:
			print('This integer is not less than 100.')
	comm.Send(value, dest=1)
	comm.Recv(value, source=size-1)
	print(value[0])

if 0 < rank < size-1:
	comm.Recv(value, source=rank-1)
	value *= rank
	comm.Send(value, dest=rank+1)

if rank == size-1:
	comm.Recv(value, source=rank-1)
	value *= rank
	# Send the value back to process 0
	comm.Send(value, dest=0)
