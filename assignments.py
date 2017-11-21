# generic solver for linear regression using numpy
import numpy as np

from regression import *
from mvn import *
from cross_validation import *
from plot import *
from error_functions import MSE
import sys
import pickle

question_features = [[[2], [1,2], [0,1,2], [0,2]], [[2,3], [1,2,3], [0,1,2,3]]]

cast = lambda x: [int(x[0]), float(x[1]), float(x[2]), float(x[3])]
raw_data = read_data("datasets/vo2-dataset.txt", cast)

ind_file = o_file="random_indices"
with open(ind_file, "rb") as istream:
    ind = pickle.load(istream)
data = random_partition(raw_data[1], 172, 1172, ind)

# for x in range(len(i_data[1])):
#     lin_r.predict(i_data[1][x]) 


################################################################################
#######################            #############################################
####################### Question 1 #############################################
#######################            #############################################

# # 1.1
# goal = select_data(data, [3]) # vector to be "learned"
# goal_t = np.append(goal[1], [])

# i_data = select_data(data, question_features[0][0]) # input features
# stats = data_statistics(i_data[0])

# dims = [2, 3, 5, 7, 10, 15]
# support = np.linspace(stats[0], stats[1], 100)
# predictors = []
# labels = []

# print("*" * 80)
# print("Number of features:{}".format(len(question_features[0][0])))

# def plot_polynomials(dims, i_data, goal, goal_t):
#     nll_table = ""
#     predictors = []
#     labels = []
#     table_string = "{0} & {1} & {2} \\\\ \\hline\n"
#     for d in dims:
#         lin_r = LinearRegression(i_data[0], d, goal[0])
#         p = lin_r.predict_vec(i_data[1])
#         print()
#         print("Degree of feature polynomials:{}.\n".format(d))
#         nll_table += table_string.format(d, lin_r.NLL, MSE(p, goal_t))
#         print("MSE ERROR:", MSE(p, goal_t))
#         print("Weights found by model", lin_r.weights)
#         print()
#         predictors.append((support, lin_r.predict_vec(support)))
#         labels.append('Polynomial order {}'.format(d))
#     print("\\begin{tabular}{|c|c|c|}")
#     print("\\hline\nPolynomial degree & NLL & MSE(Mean squared error) \\\\ \\hline")
#     print(nll_table)
#     print("\\end{tabular}")

#     return(predictors, labels)


# def iterate_over_orders(dims, i_data, goal, goal_t):
#     nll_table = ""
#     table_string = "{0} & {1:.4f} & {2:.4f} \\\\ \\hline\n"
#     for d in dims:
#         lin_r = LinearRegression(i_data[0], d, goal[0])
#         p = lin_r.predict_vec(i_data[1])
#         print()
#         # print("Degree of feature polynomials:{}.\n".format(d))
#         # print("NLL for LR:\n {0} & {1} \\\\".format(d, lin_r.NLL))
#         nll_table += table_string.format(d, lin_r.NLL, MSE(p, goal_t))
#         print("Weights found by model", lin_r.weights)
#         print()
#     print("\\begin{tabular}{|c|c|c|}")
#     print("\\hline\nPolynomial degree & NLL & MSE(Mean squared error) \\\\ \\hline")
#     print(nll_table)
#     print("\\end{tabular}")


# predictors, labels = plot_polynomials(dims, i_data, goal, goal_t)


# # plot_multiple_curves(predictors, (i_data[0], np.append(goal[0], [])), labels)


# ################################################################################
# ################################################################################

# # 1.2
# i_data = select_data(data, question_features[0][1]) # input features
# stats = data_statistics(i_data[0])
# print("*" * 80)
# print("Number of features:{}".format(len(question_features[0][1])))
# print(stats)
# iterate_over_orders(dims, i_data, goal, goal_t)

# ################################################################################
# ################################################################################

# # 1.3
# i_data = select_data(data, question_features[0][2]) # input features
# stats = data_statistics(i_data[0])
# print("*" * 80)
# print("Number of features:{}".format(len(question_features[0][2])))
# print(stats)
# iterate_over_orders(dims, i_data, goal, goal_t)

# # 1.4
# def ACSM(x):
#     return (x[2]*11.4 + 260 + x[1]*3.5) / x[1]

# acm_prediction = list()
# for i in i_data[1]:
#     acm_prediction.append(ACSM(i))

# print(MSE(acm_prediction, goal_t))
################################################################################


################################################################################
#######################            #############################################
####################### Question 2 #############################################
#######################            #############################################

# question 2.1
i_data = select_data(data, question_features[1][0]) # get data to part 1
mvn_model = MVN(i_data[0]) # compute model for mvn

stats = data_statistics(i_data[0]) # get functions to average
x = np.linspace(stats[0][0] * 0.5, 1.02 * stats[1][0], 100)
y = np.linspace(stats[0][1] * 0.5, 1.02 * stats[1][1], 100)
X, Y = np.meshgrid(x, y)
zs = np.array([mvn_model.pdf(x) for x in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
labels = ["Carga", "VO2Max", "pdf"]
plot3D(X, Y, Z, 0.001, labels)

mvn_model.print_attributes()


################################################################################
################################################################################

# question 2.2
i_data = select_data(data, question_features[1][1])
mvn_model2 = MVN(i_data[0])
mvn_model2.print_attributes()



################################################################################
################################################################################

# question 2.3
i_data = select_data(data, question_features[1][2])
mvn_model3 = MVN(i_data[0])
mvn_model3.print_attributes()


###############################################################################
######################            #############################################
###################### Question 3 #############################################
######################            #############################################
###############################################################################

# i_data = select_data(data, question_features[1][2])

# def age_partition(values, ranges, ind):
#     partitions = []
#     for i in range(len(ranges)):
#         partitions.append([])
#     for i in values:
#         for j in range(len(ranges)):
#             if(i[ind] < ranges[j][1] and i[ind] >= ranges[j][0]):
#                 partitions[j].append(i)
#     return [np.stack(i) for i in partitions] # make sure you have prorper arrays

# age_ranges = [(18,40), (40,60), (60, 200)]
# classified_data = age_partition(i_data[0], age_ranges, 0)
# test_data =  age_partition(i_data[1], age_ranges, 0)

# def muticlass_MVN(classified_data):
#     class_mvns = dict()
#     for i in range(len(classified_data)):
#         class_mvns[i] = MVN(classified_data[i])
#     return class_mvns

# mvns = muticlass_MVN(classified_data)
# for model in mvns.values():
#     print("Separated MVN`s")
#     model.print_attributes()


# from nbc import GaussianNBC

# gnbc = GaussianNBC(classified_data, 0)
# gnbc.predict(test_data[2][4])


# # operates over classified data to compute a confusion matrix
# def confusion_matrix (predictor, data):
#     conf = np.zeros((predictor.num_of_classes, predictor.num_of_classes))
#     c_i = 0
#     detections = np.zeros(2)
#     for c in data:
#         increment = 1.0 / len(c)
#         for x in c:
#             prediction = predictor.predict(x)[1]
#             conf[c_i][prediction] += increment
#             detections[int(c_i == prediction)] += 1
#         c_i += 1
#     print(detections / sum(detections))
#     return conf

# print(confusion_matrix(gnbc, test_data))



# ###############################################################################
# ######################            #############################################
# ###################### Question 4 #############################################
# ######################            #############################################
# ###############################################################################


# import gmm
# gmm_model = gmm.Supervised_GMM(classified_data, 0)
# print(confusion_matrix(gmm_model, test_data))
# gmm_model.print_params()
# crossed_data = split_data(raw_data[1], 172, 1172, [1], 3)

# #print(crossed_data[0])
# weights = d_polinomial_regression(1, crossed_data[0], crossed_data[1])
# print(weights)

# m = MVN(crossed_data[0])
# m.print_attributes()


# testing_samples = crossed_data[2][:4]
# for i in testing_samples:
#     print(m.pdf(i))
#     print(m.np_nvm(i))


# stats = data_statistics(crossed_data[0])
# print(stats)
# x = np.linspace(20, 80, 100)
# y = np.linspace(0, 180, 100)
# X, Y = np.meshgrid(x, y)
# zs = np.array([m.pdf(x) for x in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)

# plot3D(X, Y, Z, 0.001)
