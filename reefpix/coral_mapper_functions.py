#import segments_regressor

import numpy as np
import pandas as pd
import pickle
from skimage.segmentation import felzenszwalb, slic, watershed, find_boundaries
from skimage.filters import sobel
from skimage.color import rgb2gray
import random
from sklearn.metrics import r2_score, accuracy_score
import smote_cd
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings

from collections import Counter

def create_augmented_image(band):
    """
    Augments an input 2D image array by adding a border of mirrored pixels.
    
    This function takes a 2D numpy array (representing an image band) and creates a new array 
    with an additional 10-pixel border around it. The border is populated with mirrored values 
    from the original image.

    Parameters:
    -----------
    band : numpy.ndarray
        A 2D numpy array representing an image band. The input array should be of shape (height, width), with height and width being at least 10.

    Returns:
    --------
    numpy.ndarray
        A new 2D numpy array with shape (height + 20, width + 20), containing the original image 
        band in the center and a 5-pixel wide border of mirrored values from the original image.
    """
    
    if ((band.shape[0]<10) or (band.shape[1]<10)):
        raise Exception("Input array shape should be at least (10,10).")

    augmented = np.zeros((band.shape[0]+20, band.shape[1]+20), dtype=band.dtype)
    augmented[10:-10, 10:-10] = band

    augmented[:10, 10:-10] = np.flipud(band[:10, :])
    augmented[-10:, 10:-10] = np.flipud(band[-10:, :])
    augmented[10:-10, :10] = np.fliplr(band[:, :10])
    augmented[10:-10, -10:] = np.fliplr(band[:, -10:])

    augmented[:10, :10] = np.flip(band[:10, :10])
    augmented[-10:, :10] = np.flip(band[-10:, :10])
    augmented[:10, -10:] = np.flip(band[:10, -10:])
    augmented[-10:, -10:] = np.flip(band[-10:, -10:])

    return augmented


def return_superpixels_features(indexes, blue_augmented, green_augmented, red_augmented, nir_augmented, bathy_augmented=None, size_img=7):
    """
    Returns the features, i.e. the values of the pixels across all the four channels, for the given superpixel indexes.

    Parameters:
    -----------
    indexes : numpy.ndarray
        A 2D numpy array of size (n,2) representing the indices of the superpixels, with n the number of superpixels selected.
    blue_augmented : numpy.ndarray
        The augmented blue band of the image.
    green_augmented : numpy.ndarray
        The augmented green band of the image.
    red_augmented : numpy.ndarray
        The augmented red band of the image.
    nir_augmented : numpy.ndarray
        The augmented nir band of the image.
    bathy_augmented : numpy.ndarray, optional, default=None
        The augmented bathymetry of the image.
    size_img : int, optional, default=7
        The size of the superpixel.

    Returns:
    --------
    list
        A list containing the flatten values of the reflectances of all the pixels within the superpixel.
    """
    superpixels_features = []
    imin = 10-size_img//2 #here we have 10 because the augmented image is augmented by 10 pixels on each side
    imax = size_img+imin
    for i,j in indexes:
        superpixel_blue = blue_augmented[i+imin:i+imax,j+imin:j+imax]
        superpixel_green = green_augmented[i+imin:i+imax,j+imin:j+imax]
        superpixel_red = red_augmented[i+imin:i+imax,j+imin:j+imax]
        superpixel_nir = nir_augmented[i+imin:i+imax,j+imin:j+imax]
        if bathy_augmented is None:
            superpixels_features.append(np.dstack((superpixel_blue,superpixel_green,superpixel_red,superpixel_nir)).tolist())
        else:
            superpixel_bathy = bathy_augmented[i+imin:i+imax,j+imin:j+jmax]
            superpixels_features.append(np.dstack((superpixel_blue, superpixel_green, superpixel_red, superpixel_nir, superpixel_bathy)).tolist())
    return superpixels_features


def return_labels(indexes, real_rugo):
    """
    Return the labels for the given indexes from the real_rugo array.

    Parameters:
    -----------
    indexes : list of tuples
        A list of (i, j) index tuples.
    real_rugo : numpy.ndarray
        A 2D numpy array from which labels are to be extracted.

    Returns:
    --------
    numpy.ndarray
        An array of labels corresponding to the given indexes.
    """
    indexes = np.array(indexes)
    labels = real_rugo[indexes[:, 0], indexes[:, 1]]
    return labels

def random_undersampling(y,k=3):
    """
    Perform random undersampling on the k majority classes in a given 2D array of compositional vector labels.
    
    This function identifies the k majority classes in the input 2D array `y`, where each class label is a compositional vector, and reduces their sample counts to match the (k+1)-th largest class.

    Parameters:
    -----------
    y : array-like, shape (n_samples, n_features)
        The input 2D array of class labels, where each row is a compositional vector representing a class.
    k : int, optional, default=3
        The number of majority classes to undersample.

    Returns:
    --------
    indexes_to_delete : list
        A list of indexes corresponding to the samples that should be deleted to achieve the undersampling.

    Notes:
    ------
    - The function uses `random.sample` to randomly select which samples to delete from the majority classes.
    - The input array `y` should be a 2D array of class labels.
    - The function assumes there are at least k+1 unique compositional vectors in `y`.

    """
    classes , count = np.unique(y,return_counts=True)
    sorted_classes = sorted(classes, key=lambda k: count[k],reverse=True)
    sorted_count = sorted(count,reverse=True)
    threshold = sorted_count[k]
    indexes_to_delete = []
    for i in range(k):
        indexes_to_delete_temp = random.sample(list(np.where(np.array(y)==sorted_classes[i])[0]),sorted_count[i]-threshold)
        indexes_to_delete += indexes_to_delete_temp
    return indexes_to_delete



def percentage_occurence(l,n):
    """
    Given a list l, returns the percentage of occurences of each value n (from 0 to n-1).
    """
    l_return=np.zeros(n)
    counter_l = Counter(l)
    for value, count in counter_l.items():
        l_return[value] = count
    return l_return/l_return.sum()


def create_labels_dataset(segments,zones,nb_zones=0):
    """
    Create a dataset of labels for image segments based on the zones they overlap.

    This function takes as input a 2D array of image segments and a 2D array of zones, and generates a dataset of labels for each segment. Each label represents the percentage occurrence of each zone within the segment.

    Parameters:
    -----------
    segments : array-like, shape (height, width)
        The 2D array representing the segmented image. Each unique value corresponds to a different segment.
    zones : array-like, shape (height, width)
        The 2D array representing different zones in the image. Each unique value corresponds to a different zone.
    nb_zones : int, optional, default=0
        The number of unique zones. If set to 0, the function automatically determines the number of zones based on the unique values in the `zones` array.

    Returns:
    --------
    labels_segments : array, shape (n_segments, nb_zones)
        A 2D array where each row corresponds to a segment and each column represents the percentage occurrence of a specific zone within that segment.

    Example:
    --------
    >>> segments = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
    >>> zones = np.array([[1, 1, 2], [0, 0, 2], [1, 1, 2]])
    >>> create_labels_dataset(segments, zones)
    array([[0.5, 0.5, 0. ],[0. , 0.4, 0.6]])

    Notes:
    ------
    - The `zones` and `segments` arrays should have the same shape.
    - The `percentage_occurence` function is used to compute the percentage of each zone within each segment.
    """
    
    if segments.shape != zones.shape:
        raise Exception(f"The arrays segments and zones must have the same shape, but are of shape {segments.shape} and {zones.shape}.")

    if nb_zones==0:
        nb_zones=len(np.unique(zones))
        
    labels_segments=[]
    zones_flat = zones.flatten()
    flat_seg = segments.flatten()
    
    idx_sort = np.argsort(flat_seg)
    sorted_segments = flat_seg[idx_sort]
    vals, idx_start, count = np.unique(sorted_segments, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start[1:]) # splits the indices into separate arrays
    
    for list_indexes_superpixel in res:
        list_zones=zones_flat[list_indexes_superpixel]
        labels_segments.append(percentage_occurence(list_zones,nb_zones))
    return np.array(labels_segments)


####################


def create_map_from_segments(segments,segments_pred):
    """ 
    Creates the map from the given segments and their associated labels.
    
    Parameters:
    -----------
    segments : array-like, shape (height, width)
        The 2D array representing the segmented image. Each unique value corresponds to a different segment.
    segments_pred : list, length (number of segments)
        The list of the majority classes within each segments.
        
    Returns:
    --------
    array, shape (height, width)
        A 2D array where each value (i,j) corresponds to the class of the segment of index segments[i,j].

    Example:
    --------
    >>> segments = np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]])
    >>> labels_segments = np.array([0, 2])
    >>> create_map_from_segments(segments, labels_segments)
    array([[0., 0., 2.], [0., 0., 2.], [2., 2., 2.]])
    """
    created_map = np.zeros(np.shape(segments))
    segments_pred = np.array(segments_pred)
    
    idx_sort = np.argsort(segments_pred)
    sorted_segments = segments_pred[idx_sort]
    vals, idx_start, count = np.unique(sorted_segments, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start[1:])
    
    for list_indexes_zone in res:
        created_map[np.isin(segments,list_indexes_zone)] = segments_pred[list_indexes_zone[0]]
    return(created_map)


####################

        
def score_map(true_map, predicted_map, classes_to_remove=[], segments=None, train_indexes_pb=None, train_indexes_ob=None, size_superpixel=7):
    """
    Computes the pixelwise score of a predicted map. 
    
    The functions compares the predicted map and with the true map and counts the number of correctly predicted pixels and divides by the total number of pixels. The function allows to mask some pixels in the process, for instance some classes that may not be of interested, or the pixels that have been used during the training of the model.
    
     Parameters:
    -----------
    true_map : array-like, shape (height, width)
        The 2D array representing the ground-truth map.
    predicted_map : array-like, shape (height, width)
        The 2D array representing the predicted map.
    classes_to_remove : list, optional, default=[]
        The list of the classes to remove, i.e. the classes (from the true map) that must not be counted in the score.
    segments : array-like, shape (height, width), optional, default=None
        The 2D array representing the segmented image. Each unique value corresponds to a different segment. Only used when train_indexes_ob is specified.
    train_indexes_pb : list, optional, default=None
        The list of the indexes of the pixels that were used to train the pixel-based model.
    train_indexes_ob : list, optional, default=None
        The list of the indexes of the segments that were used to train the object-based model.
    size_superpixel : int, optional, default=7
        The size of the superpixels in the pixel-based model.
        
    Returns:
    --------
    float
        The accuracy (pixelwise score) of the predicted map.
    """
    if true_map.shape != predicted_map.shape:
        raise Exception("The true and predicted maps must have the same shape.")
        
    if train_indexes_pb is None:
        true = true_map[np.isin(true_map,classes_to_remove,invert=True)]
        predicted = predicted_map[np.isin(true_map,classes_to_remove,invert=True)]
        acc = sum(true == predicted) / np.prod(np.shape(true))
    else:
        # masking the training values of the pixel-based
        k = size_superpixel-1
        pb_training_mask = np.ones(np.shape(true_map))
        n_rows, n_cols = np.shape(true_map)
        for i, j in train_indexes_pb:
            # calculate the bounds of the surrounding region
            row_start = max(i - k, 0)
            row_end = min(i + k + 1, n_rows)
            col_start = max(j - k, 0)
            col_end = min(j + k + 1, n_cols)
            # set the mask to 0 in the surrounding region
            pb_training_mask[row_start:row_end, col_start:col_end] = 0
            
        if train_indexes_ob is None:
            true_interest_zone = true_map[np.isin(true_map,classes_to_remove,invert=True)]
            predicted_interest_zone = predicted_map[np.isin(true_map,classes_to_remove,invert=True)]
            pb_training_mask = pb_training_mask[np.isin(true_map,classes_to_remove,invert=True)]
        else:
            true_no_train = true_map[np.isin(segments,train_indexes_ob,invert=True)]
            true_interest_zone = true_no_train[np.isin(true_no_train,classes_to_remove,invert=True)]
            predicted_no_train = predicted_map[np.isin(segments,train_indexes_ob,invert=True)]
            predicted_interest_zone = predicted_no_train[np.isin(true_no_train,classes_to_remove,invert=True)]
            pb_training_mask = pb_training_mask[np.isin(segments,train_indexes_ob,invert=True)]
            pb_training_mask = pb_training_mask[np.isin(true_no_train,classes_to_remove,invert=True)]
        acc = np.sum(np.logical_and(pb_training_mask,true_interest_zone==predicted_interest_zone))/np.sum(pb_training_mask)
        
    return(acc)