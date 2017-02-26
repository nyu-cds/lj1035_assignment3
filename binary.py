"""
This program generates all binary strings of length n that contain k zero bits.
"""
from itertools import combinations


def zbits(n, k):
    """
    :param n: length of string
    :param k: number of zero bits
    :return: a set of strings
    """
    # initialize the set
    strings = set()
    position_index = range(0, n)
    # loop over the iterator
    for item in combinations(position_index, k):
        # create a list of n '1's
        string = ['1'] * len(position_index)
        # loop over the elements in each item
        for element in item:
            string[element] = '0'
        # join the elements into a single string
        strings.add(''.join(string))
    return strings


def main():
    # tests to ensure program correctness
    assert zbits(4, 3) == {'0100', '0001', '0010', '1000'}
    assert zbits(4, 1) == {'0111', '1011', '1101', '1110'}
    assert zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}

if __name__ == '__main__':
    main()