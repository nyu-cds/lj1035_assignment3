from pyspark import SparkContext
from operator import add
from math import sqrt


if __name__ == '__main__':
    sc = SparkContext("local", "average square root")
    
    nums = sc.parallelize(range(1, 1001))
    result = nums.map(lambda x: sqrt(x)) \
                 .fold(0, add) / 1000

    print(result)
