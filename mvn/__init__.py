import numpy as np
from scipy.stats import multivariate_normal
import scipy.integrate as integrate

__all__ = ["MVN", "MultiClassMVN"]


class MVN:


    # initialize all variables to compute probability
    def __init__(self, data):
        self.dimensions = len(data[0]) # number of feature dimensions
        self.sample_size = len(data)

        self.mean = np.mean(data, axis=0) # empirical mean
        cov = self.covariance_matrix(data)

        # numerical constants due to normalization
        self.cov_det = np.linalg.det(cov)
        norm_coef = (self.cov_det * ((2 * np.pi) ** (self.dimensions))) ** 0.5
        self.norm_coef = 1 / norm_coef
        self.precision = np.linalg.inv(cov) # empirical precision matrix
        self.cov = cov

        # compute conditional parameters
        self.predict_params()


    # compute covariance matrix
    def covariance_matrix(self, data):
        cov =  np.dot(data.T, data)
        cov = (cov / self.sample_size) - np.einsum('i,j->ij', self.mean, self.mean)
        return cov


    # compute multivariate normal pdf
    def pdf(self, x):
        return self.norm_coef * np.exp(-0.5 * 
            np.dot((x - self.mean),
                np.dot(self.precision, x - self.mean)))



    # conditional pdfs
    def conditional_pdf(self, x):
        return self.norm * np.exp(-((x[-1] - self.conditional_mean(x)) ** 2) / (2 * self.variance))


    # testing function using numpy MVN model
    def np_nvm(self, x):
        return multivariate_normal.pdf(x, self.mean, np.linalg.inv(self.precision))


    # print MLE statistics used to build normal
    def print_attributes(self):
        print("Precision Matrix:\n", self.precision)
        print("Covariance Matrix:\n", np.linalg.inv(self.precision))
        print("Mean:\n", self.mean)



    def predict_params(self):
        sig11 = self.cov[0: self.dimensions - 1, 0: self.dimensions - 1]
        sig12 = self.cov[-1][0:self.dimensions - 1]
        self.scale = np.dot(sig12, np.linalg.inv(sig11))
        self.variance = self.cov[-1][-1] - np.dot(self.scale, sig12)
        self.norm = 1 / ((2 * np.pi * self.variance) ** 0.5)


    def conditional_mean(self, x):
        return self.mean[-1] + np.dot(self.scale, self.mean[0: -1] - x[0: -1])


    # gaussian prediction
    def predict(self, x):
        # compute conditional dist
        mu = self.conditional_mean(x)
        # pdf = 
        # pdf * np.exp(-0.5 * ((x[-1] - mu) ** 2) * self.conditional_std)
        return [mu, (mu + (self.variance ** 0.5), mu - (self.variance ** 0.5))]


    def classify(self, x, ranges, remove_class=True):
        scores = list()
        if remove_class:
            ind = 1
        else:
            ind = 0
        for i in range(len(ranges)):
            scores.append((integrate.quad(lambda y: self.pdf(np.append([y], x[ind:])), 
                ranges[i][0], ranges[i][1]), i))
        return max(scores)

class MultiClassMVN:

    def __init__(self, classified_data):
        class_mvns = dict()
        for i in range(len(classified_data)):
            class_mvns[i] = MVN(classified_data[i])
        self.class_mvns = class_mvns
        self.num_of_classes = len(self.class_mvns.values())

    def predict(self, x):
        scores = []
        for c in self.class_mvns.keys():
            scores += [(self.class_mvns[c].pdf(x), c)]
        return max(scores)
