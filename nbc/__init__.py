import numpy as np

__all__ = ["NBC"]


class GaussianNBC:


    def extract_features(self, x):
        return np.append(x[:self.class_label], x[self.class_label + 1:])

    # function to compute the most probable class using Gaussian Likelihood
    def predict(self, x):
        scores = list()
        for i in range(self.num_of_classes):
            scores.append((self.class_likelihood(x, i), i))
        return max(scores)

    def __init__(self, classified_data, class_label):
        self.num_of_classes = len(classified_data)
        self.num_of_features = len(classified_data[0][0]) - 1
        self.params = np.zeros((self.num_of_classes, self.num_of_features, 2))

        feature_ind = list(range(len(classified_data[0][0])))
        del feature_ind[class_label]
        feature_map = dict()
        for f in range(len(feature_ind)):
            feature_map[feature_ind[f]] = f

        # compute empirical mean and std deviation for each feature 
        #  on each class
        for c in range(self.num_of_classes):
            h_data = classified_data[c].T # transpose to make easier
            for f in feature_ind:
                offseted_ind = feature_map[f]
                self.params[c][offseted_ind][0] = np.mean(h_data[f], axis=0)
                self.params[c][offseted_ind][1] = np.std(h_data[f], axis=0)

        self.class_label = class_label


    # compute a unidimensinal Gaussian at X defined by the [c][f] params
    def normal(self, c, f, x):
        mean, std = self.params[c][f]
        return np.exp(-0.5 * (((x - mean) / std) ** 2)) \
            / (((np.pi * 2) ** 0.5) * std)


    # compute the likilehoodd for a class following the NBC assumption
    def class_likelihood(self, x, c):
        l = 1.0
        features = self.extract_features(x)
        for i in range(len(features)):
            l = l * self.normal(c, i, x[i])
        return l
