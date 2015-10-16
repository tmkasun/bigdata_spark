__author__ = 'tmkasun'

from pyspark import SparkConf, SparkContext
import numpy as np
from operator import add
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Matrices


# dv1 = np.array([1, 2, 3, 4, 5])
# dv2 = Vectors.sparse(7, (0, 1, 2, 3, 5), (2.3, 23.24, 123.213, 221, 21))
# print(dv2)
# lp = LabeledPoint(3, [3, 4, 5, 6])
#
# dm = Matrices.dense(2, 3, [12, 34, 11.23, 4.54, 23.54, 76.45])
# print(dm)

# Create a Scala Spark Context.
conf = SparkConf().setMaster('local').setAppName("Knnect_app")
sc = SparkContext(conf=conf)

# Load our input data.
lines = sc.textFile('./topten-2.txt')
data = sc.parallelize([x for x in range(60)])
# Split it up into words.
# words = lines.flatMap(lambda line: line.split(','))
# Transform into pairs and count.
# counts = words.map(lambda word: (word, 1)).reduceByKey(add).collect()
fltrd = data.filter(lambda x: x>10)

print(fltrd.count())
