# generic solver for linear regression using numpy
import numpy as np

__all__ = ["LinearRegression"]



class LinearRegression():

    def __init__(self, data, d, goal):
        self.feature_length = len(data[0])
        self.d = d 
        basis = self.d_polinomial_basis(data) # column space basis
    
        # compute X^TX
        t = np.transpose(basis) # get basis for row space
        feature_spam = np.linalg.inv(np.dot(t, basis))
        weights = np.dot(feature_spam, np.dot(t, goal)) # solve system for weights
        self.weights = np.append(weights, [])



    # transform an input vector of features into polinomial basis
    def d_polinomial_vector(self, x):
        aux = np.append(x,[]) # make sure x is an array
        feature_length = len(aux)
        basis = np.ones(1 + (feature_length * self.d))
        for i in range(0 , feature_length): # through features
            for j in range(1, self.d + 1): # through exponents
                    basis[(i * self.d) + j - 1] = aux[i - 1] ** j
        return np.array(basis)

    def predict(self, x):
        return np.dot(self.weights, self.d_polinomial_vector(x))

    def predict_vec(self, x):
        p = np.zeros(len(x))
        for i in range(len(x)):
            p[i] = self.predict(x[i])
        return p

# def d_polinomial_basis(d, feature_length, data):
#     basis = np.ones((len(data), 1 + (feature_length * d)))
#     for k in range(len(data)): # over all data points
#         for i in range(0 , feature_length): # through features
#             for j in range(1, d + 1): # through exponents
#                 basis[k][(i * d) + j - 1] = data[k][i - 1] ** j
#     return basis

    def d_polinomial_basis(self, data):
        basis = np.ones((len(data), 1 + (self.feature_length * self.d)))
        for k in range(len(data)): # over all data points
            for i in range(0 , self.feature_length): # through features
                for j in range(1, self.d + 1): # through exponents
                    basis[k][(i * self.d) + j - 1] = data[k][i - 1] ** j
        return basis


# # solve regression system w = (XˆT X)ˆ(-1)(XˆT)b
# def solve_regression(data, phi, goal):
#     basis = phi(len(data[0]), data) # column space basis
    
#     # compute X^TX
#     t = np.transpose(basis) # get basis for row space
#     feature_spam = np.linalg.inv(np.dot(t, basis))
#     weights = np.dot(feature_spam, np.dot(t, goal)) # solve system for weights
#     return weights


# def d_polinomial_basis(d, feature_length, data):
#     basis = np.ones((len(data), 1 + (feature_length * d)))
#     for k in range(len(data)): # over all data points
#         for i in range(0 , feature_length): # through features
#             for j in range(1, d + 1): # through exponents
#                 basis[k][(i * d) + j - 1] = data[k][i - 1] ** j
#     return basis

# # def d_polinomial_basis(d, feature_length, data):
# #     fun = np.vectorize(d_polinomial_vector)
# #     return fun(data, d)


# # transform an input vector of features into polinomial basis
# def d_polinomial_vector(x, d):
#     aux = np.append(x,[]) # make sure x is an array
#     feature_length = len(aux)
#     basis = np.ones(1 + (feature_length * d))
#     for i in range(0 , feature_length): # through features
#         for j in range(1, d + 1): # through exponents
#                 basis[(i * d) + j - 1] = aux[i - 1] ** j
#     return basis


# # encapsulate polinomial functions as basis
# def d_polinomial_regression(d, data, goal):
#     phi = lambda x, y: d_polinomial_basis(d, x, y)
#     return solve_regression(data, phi, goal)
