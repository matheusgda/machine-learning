import numpy as np
import mvn

class Supervised_GMM:


    def extract_features(self, x):
        return np.append(x[:self.class_label], x[self.class_label + 1:])


    def clean_data(self, data):
        return map(lambda x: np.apply_along_axis(self.extract_features, 1,  x),
            data)

    # function to compute the most probable class using Gaussian Likelihood
    def predict(self, x):
        scores = list()
        for i in range(self.num_of_classes):
            scores.append((self.class_likelihood(x, i), i))
        return max(scores)


    def __init__(self, classified_data, class_label):
        self.num_of_classes = len(classified_data)
        self.class_label = class_label
        self.mixture_models = list(map(lambda c: mvn.MVN(c) , 
            self.clean_data(classified_data)))
        priors = np.zeros(self.num_of_classes)
        self.class_label = class_label

        samples = 0
        for c in range(self.num_of_classes):
            members = len(classified_data[c])
            priors[c] += members
            samples += members
        self.priors = priors / samples


    # compute class likelihood using gmm priors
    def class_likelihood(self, x, c):
        features = self.extract_features(x)
        return self.priors[c] * self.mixture_models[c].pdf(features)


    def print_params(self):
        for c in range(self.num_of_classes):
            print("Parameters for class {0}".format(c))
            self.mixture_models[c].print_attributes()
            print("Prior = {0}".format(self.priors[c]))