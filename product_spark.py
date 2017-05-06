from pyspark import SparkContext
from operator import mul


if __name__ == '__main__':
    sc = SparkContext("local", "product")
    
    nums = sc.parallelize(range(1, 1001))
    result = nums.fold(1, mul)

    print(result)
