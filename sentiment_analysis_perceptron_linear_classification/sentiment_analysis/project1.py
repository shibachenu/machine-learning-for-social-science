import math
from string import punctuation, digits
import numpy as np
import random

# Part I


def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices


def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    """
    print("Hinge loss for x_feature vector: " + str(feature_vector) + ", label: " + str(label) + ", theta: " + str(
        theta) + ", theta_0: " + str(theta_0))
    prediction = np.matmul(feature_vector, theta) + theta_0
    agreement = label * prediction
    loss = max(0, 1 - agreement)
    print("Hinge loss is: " + str(loss))
    return loss

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    prediction = np.matmul(feature_matrix, theta) + theta_0
    agreement = np.multiply(labels, prediction)
    loss_all = np.maximum(0, 1 - agreement)
    loss = np.average(loss_all)
    print("Hinge loss is: " + str(loss))
    return loss


def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    debug = False
    if debug: print("Before update: theta: " + str(current_theta) + ", theta0: " + str(current_theta_0))
    assignment = label * (np.dot(current_theta, feature_vector) + current_theta_0)
    theta = current_theta
    theta0 = current_theta_0
    if assignment <= 0:
        theta = label * feature_vector.transpose() + current_theta
        theta0 = label + current_theta_0
    if debug: print("After update: theta: " + str(theta) + ", theta0: " + str(theta0))
    return theta, theta0

def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    # Your code here
    p = feature_matrix.shape[1]
    theta = np.zeros(p)
    theta0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            feature_vector = feature_matrix[i, :]
            label = labels[i]
            theta, theta0 = perceptron_single_step_update(feature_vector, label, theta, theta0)
    return theta, theta0


def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    p = feature_matrix.shape[1]
    n = feature_matrix.shape[0]
    theta = np.zeros(p)
    theta0 = 0

    theta_sum = theta
    theta0_sum = theta0

    for t in range(T):
        for i in get_order(n):
            feature_vector = feature_matrix[i]
            label = labels[i]
            theta, theta0 = perceptron_single_step_update(feature_vector, label, theta, theta0)
            theta_sum = theta_sum + theta
            theta0_sum = theta0_sum + theta0

    theta_average = theta_sum / (n * T)
    theta0_average = theta0_sum / (n * T)
    return theta_average, theta0_average

def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    updated_theta = current_theta
    updated_theta0 = current_theta_0

    assignment = label * (np.dot(feature_vector, current_theta) + current_theta_0)

    if (assignment <= 1):
        updated_theta = (1 - eta * L) * current_theta + eta * label * feature_vector
        updated_theta0 = current_theta_0 + eta * label
    else:
        updated_theta = (1 - eta * L) * current_theta
        updated_theta0 = current_theta_0

    return updated_theta, updated_theta0


def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    p = feature_matrix.shape[1]
    theta = np.zeros(p)
    theta0 = 0
    count = 1
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            feature_vector = feature_matrix[i]
            label = labels[i]
            eta = 1 / (count ** 0.5)
            theta, theta0 = pegasos_single_step_update(feature_vector, label, L, eta, theta, theta0)
            count = count + 1
    return theta, theta0

# Part II

def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    assignment = np.matmul(feature_matrix, theta) + theta_0

    def prediction(row):
        return 1 if row>0.0000002 else -1

    vprediction = np.vectorize(prediction)
    classification = vprediction(assignment)
    return classification

def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the validation
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """

    theta, theta0 = classifier(train_feature_matrix, train_labels, **kwargs)
    train_prediction = classify(train_feature_matrix, theta, theta0)
    train_accuracy_raw = train_prediction == train_labels
    train_accuracy = np.mean(train_accuracy_raw)

    val_predictions = classify(val_feature_matrix, theta, theta0)
    val_accuracy_raw = val_predictions == val_labels
    val_accuracy = np.mean(val_accuracy_raw)

    return train_accuracy, val_accuracy


def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()

import re
def get_stop_words():
    with open('stopwords.txt') as f:
        stopwords = re.findall("[a-zA-Z\-\.'/]+", f.read())
    return stopwords

def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    dictionary = {} # maps word to unique index

    stop_words = get_stop_words()

    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if (word not in dictionary and word not in stop_words):
                dictionary[word] = len(dictionary)
    return dictionary
    #before stop words: 13234
    #after stop words removed: 13108


def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here

    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] += 1
    return feature_matrix


def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
