from mpi4py import MPI
comm = MPI.COMM_WORLD
# Get the rank of the process within the communicator
rank = comm.Get_rank()

# Print "Hello" when rank is even
if rank % 2 == 0:
	print('Hello from process', rank)
# Print "Goodbye" when rank is odd
if rank % 2 != 0:
	print('Goodbye from process', rank)

