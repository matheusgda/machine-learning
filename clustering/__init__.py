import numpy as np

__all__ = ["IterativeClustering"]


# this is to be used as a base class
class IterativeClustering:

    def __init__(self, error):
        self.accepted_training_error = error


    def compute_clusters(self, clusters, data, init=None):
        self.clusters = clusters
        if init is None:
            self.random_init(data, clusters)
        else:
            self.prototypes = init

        error = np.inf
        proceed = True
        assignments = np.array([])
        while proceed:
            e2, assignments = self.expect(data)
            self.optimize(data, assignments)
            proceed = self.proceed(error, e2)
            error = e2
        return assignments


    # E-step
    def expect(self, data):
        return None


    # M-step
    def optimize(self, data):
        return None


    # to make stop criteria parametrizable
    def proceed(self, e1, e2):
        # return e1 > self.accepted_training_error and e1 != e2
        return e1 != e2


    # initialization with 
    def random_init(self, data, clusters):
        self.prototypes = data[np.random.random_integers(0, len(data), clusters)]



class KMeans(IterativeClustering):

    def __init__(self):
        super(KMeans, self).__init__(0.0)


    def expect(self, data):
        assignments = np.zeros(len(data), dtype=np.uint32)
        acceptance = 0
        for x in range(len(data)):
            dist = np.inf
            for k in range(self.clusters):
                dist_to_c = np.linalg.norm(data[x] - self.prototypes[k])
                if dist_to_c < dist: # update the best assignment
                    dist = dist_to_c
                    assignments[x] = k
            acceptance += dist # update overall  metric for clustering
        return (acceptance, assignments)


    # update cluster prototypes using assigned sample average
    def optimize(self, data, assignments):
        prototypes = np.zeros((len(self.prototypes), len(data[0])))
        counters = np.zeros(len(self.prototypes))
        for i in range(len(data)):
            prototypes[assignments[i]] += data[i]
            counters[assignments[i]] += 1
        for i in range(len(prototypes)):
            self.prototypes[i] = prototypes[i] / counters[i]


#class EMGMM(IterativeClustering)
