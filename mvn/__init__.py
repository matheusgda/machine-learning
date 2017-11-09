import numpy as np
from scipy.stats import multivariate_normal

__all__ = ["mvn"]

class mvn:


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


    # testing function using numpy mvn model
    def np_nvm(self, x):
        return multivariate_normal.pdf(x, self.mean, np.linalg.inv(self.precision))


    # print MLE statistics used to build normal
    def print_attributes(self):
        print("Precision Matrix:\n", self.precision)
        print("Covariance Matrix:\n", np.linalg.inv(self.precision))
        print("Mean:\n", self.mean)
