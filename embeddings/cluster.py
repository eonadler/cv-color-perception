import os
import numpy as np
from sklearn.cluster import KMeans
from numbers import Number
from pandas import DataFrame
import sys, codecs, numpy
import json
import pickle
import pickle5 as pickle5
from shutil import copyfile


class autovivify_list(dict):
#    '''A pickleable version of collections.defaultdict'''
    def __missing__(self, key):
#    '''Given a missing key, set initial value to an empty list'''
        value = self[key] = []
        return value

    def __add__(self, x):
#    '''Override addition for numeric types when self is empty'''
        if not self and isinstance(x, Number):
            return x
        raise ValueError

    def __sub__(self, x):
#    '''Also provide subtraction method'''
        if not self and isinstance(x, Number):
            return -1 * x
        raise ValueError
        

def load_data(path):
    with open(path, 'rb') as fp:
        data = pickle5.load(fp,encoding= 'unicode_escape')
    return data


def get_all_labels(data):
    labels = []
    for img in data.keys():
        labels.append(img[:])
    return labels


def build_word_vector_matrix(data, labels):
    '''Return the vectors and labels for the first n_words in vector file'''
    numpy_arrays = []
    labels_array = []
    for key in data.keys():#data.keys():
        labels_array.append(key)
        numpy_arrays.append(np.array(data[key]))
    return np.array(numpy_arrays), labels_array


def get_kmeans_model(labels, reduction_factor=0.014):
    input_vector_file = sys.argv[1]
    n_labels = int(len(labels))
    reduction_factor = float(reduction_factor)
    n_clusters = int(n_labels*reduction_factor)
    df, labels_array = build_word_vector_matrix(input_vector_file, n_labels, labels)
    kmeans_model = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
    kmeans_model.fit(df)
    return kmeans_model


def find_label_clusters(labels_array, cluster_labels):
    '''Return the set of labels in each cluster'''
    cluster_to_labels = autovivify_list()
    for c, i in enumerate(cluster_labels):
        cluster_to_labels[i].append(labels_array[c])
    return cluster_to_labels


def write_clusters(kmeans_model, labels_array, image_data_path, directory_name, output_path='./'):
    ###
    cluster_labels  = kmeans_model.labels_
    cluster_inertia   = kmeans_model.inertia_
    cluster_to_labels  = find_label_clusters(labels_array, cluster_labels)
    ###
    os.chdir(output_path)
    os.mkdir(directory_name)
    ###
    for j,c in enumerate(cluster_to_labels):
        os.mkdir(directory_name+'/cluster{}'.format(j))
        for label in cluster_to_words[c]:
            copyfile(image_data_path+'/{}'.format(label), directory_name'/cluster{}/{}'.format(j,label))
    ###
    return