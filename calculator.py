# -----------------------------------------------------------------------------
# calculator.py
#
# Speedup: 2.677 / 0.035 = 76.5x
#
# Original:
# By cPython: 1000014 function calls in 2.677 seconds
# By Line_profiler: I found that most of the runtime were spent on the 'for'
#                   loops in functions add(), multiply(), and sqrt().
# Therefore, I decided to optimize the program by using np.add(), np.multiply()
# and np.sqrt() instead.
#
# After optimization:
# By cPython: 4 function calls in 0.035 seconds
# -----------------------------------------------------------------------------
import numpy as np


def hypotenuse(x, y):
    """
    Return sqrt(x**2 + y**2) for two arrays, a and b.
    x and y must be two-dimensional arrays of the same shape.
    """
    xx = np.multiply(x, x)
    yy = np.multiply(y, y)
    zz = np.add(xx, yy)
    return np.sqrt(zz)