from mpi4py import MPI
from parallel_sorter import generate_array, slice_array, sort_data
import numpy
import unittest

class Test(unittest.TestCase):

    def setUp(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
    
    def test_generate_array(self):
        """
        Unit test for the generate_array function. 
        """
        self.assertEqual(generate_array().shape, (10000,))

    def test_slice_array(self):
        """
        Unit test for the slice_array function.
        """
        input_arr = numpy.array([2, 3, 9, 0, 4, 5, 1, 8, 7, 6])
        sliced_arr = slice_array(input_arr, 4)
        self.assertEqual(sliced_arr, [[0], [2, 3, 1], [4, 5, 6], [9, 8, 7]])

    def test_sort_data(self):
        """
        Unit test for the sort_data function.
        """
        sorted_data = sort_data()
        if self.rank == 0:
            self.assertTrue((sorted(sorted_data) == sorted_data).all())
        else:
            self.assertEqual(None, sorted_data)


if __name__ =='__main__':
    unittest.main()