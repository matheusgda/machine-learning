import numpy as np

from cross_validation import *
from plot import *
from error_functions import MSE
import sys
import pickle
import clustering

cast = lambda x: [int(x[0]), float(x[1]), float(x[2]), float(x[3])]
raw_data = read_data("datasets/vo2-dataset.txt", cast)

kmeans = clustering.KMeans()
classified_data = kmeans.compute_clusters(3, raw_data[1])
print(kmeans.prototypes)

classified_data = kmeans.compute_clusters(4, raw_data[1])
print(kmeans.prototypes)