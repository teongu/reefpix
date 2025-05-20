import numpy as np
import importlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.segmentation import mark_boundaries
from skimage.segmentation import find_boundaries
import time
import pandas as pd
import random
import matplotlib.colors as colors
from scipy.stats import skew, kurtosis

import warnings

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


####################

def neighbors_indexes(i,j,imax,jmax):
    """
    Return the indexes of the neighbors of a given index (left, right, above, below).
    
    Parameters
    ----------
    i : int
        The index of the y-axis (the rows).
    j : int
        The index of the x-axis (the columns).
    imax : int
        The maximal index of the y-axis (the rows).
    jmax : int
        The maximal index of the x-axis (the columns).
    
    Returns
    -------
    list, shape (p,2)
        The indexes of the neighbors, 2<=p<=4 depending where the input index is located. 
    """
    if i==0:
        if j==0:
            ind_neighbors = [[i+1,j],[i,j+1]]
        elif j==jmax:
            ind_neighbors = [[i+1,j],[i,j-1]]
        else:
            ind_neighbors = [[i+1,j],[i,j-1],[i,j+1]]
    elif i==imax:
        if j==0:
            ind_neighbors = [[i-1,j],[i,j+1]]
        elif j==jmax:
            ind_neighbors = [[i-1,j],[i,j-1]]
        else:
            ind_neighbors = [[i-1,j],[i,j-1],[i,j+1]]
    elif j==0:
        ind_neighbors = [[i-1,j],[i+1,j],[i,j+1]]
    elif j==jmax:
        ind_neighbors = [[i-1,j],[i+1,j],[i,j-1]]
    else:
        # usual case with 4 neighbors
        ind_neighbors = [[i-1,j],[i+1,j],[i,j-1],[i,j+1]]
    return(ind_neighbors)


def find_adjacent_segments(segments, boundaries):
    """
    For each segment, find the neighboring pixels and the segments they are associated to.
    
    Parameters
    ----------
    segments : array-like, shape (p,q)
        The map of the segments, where the value of each pixel is the segment it belongs to.
    boundaries : array-like, shape (p,q)
        The boolean array where the pixels on the boundaries are marked as True, and the others as False.
    
    Returns
    -------
    list of list, length n
        The list where the i-th element contains the adjacents segments of the i-th segment.
    list of list, length n
        The list where the i-th element contains the weights of the adjacents segments of the i-th segment.
    """
    segments=np.array(segments)
    boundaries=np.array(boundaries)
    
    imax = np.shape(segments)[0]-1
    jmax = np.shape(segments)[1]-1

    segments_neighbors = [[] for _ in np.unique(segments)]

    boundaries_x, boundaries_y = np.where(boundaries==True)
    for i,j in zip(boundaries_x, boundaries_y):
        num_segment = segments[i,j]
        ind_neighbors = neighbors_indexes(i,j,imax,jmax)
        for i_neighbor,j_neighbor in ind_neighbors:
            neighbor_seg = segments[i_neighbor,j_neighbor]
            if neighbor_seg != num_segment:
                segments_neighbors[num_segment].append(neighbor_seg)
                
    # we sum all the neighboring pixels that belong to the same segment
    adjacent_segments = []
    adjacent_segments_weights = []
    for s in segments_neighbors:
        unique_values, weights = np.unique(s,return_counts=True)
        adjacent_segments.append(unique_values)
        adjacent_segments_weights.append(weights/np.sum(weights))
    
    return(adjacent_segments, adjacent_segments_weights)


##################################################

def create_features_old(segments, B_band, G_band, R_band, NIR_band):
    """
    Create the features, based on the statistics values of the reflectances of each segment.
    
    Parameters
    ----------
    segments : array-like, shape (p,q)
        The map of the segments, where the value of each pixel is the segment it belongs to.
    B_band : array-like, shape (p,q)
        The blue band of the image.
    G_band : array-like, shape (p,q)
        The green band of the image.
    R_band : array-like, shape (p,q)
        The red band of the image.
    NIR_band : array-like, shape (p,q)
        The NIR band of the image.
    
    Returns
    -------
    pandas.DataFrame, shape (n,16)
        The DataFrame containing the 16 features for each of the n segments.
    """
    if (np.shape(NIR_band)!=np.shape(R_band)) or (np.shape(R_band)!=np.shape(G_band)) or (np.shape(G_band)!=np.shape(B_band)):
        raise Exception('The dimensions of the different bands are not equal.')
        
    Bband = np.array(B_band)
    Gband = np.array(G_band)
    Rband = np.array(R_band)
    NIRband = np.array(NIR_band)
    segments_ = np.array(segments)
        
    statistics_df=pd.DataFrame(columns=['mean_blue','var_blue','skew_blue','kurt_blue',
                                        'mean_green','var_green','skew_green','kurt_green',
                                       'mean_red','var_red','skew_red','kurt_red',
                                       'mean_nir','var_nir','skew_nir','kurt_nir'])
    
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation due to catastrophic cancellation.")
        for index_superpixel in np.unique(segments_):
            if index_superpixel>=0:
                x,y=np.where(segments_==index_superpixel)
                b_values=Bband[x,y]
                g_values=Gband[x,y]
                r_values=Rband[x,y]
                nir_values=NIRband[x,y]
                statistics_df.loc[index_superpixel]=[np.mean(b_values),np.var(b_values),skew(b_values),kurtosis(b_values),
                                                    np.mean(g_values),np.var(g_values),skew(g_values),kurtosis(g_values),
                                                    np.mean(r_values),np.var(r_values),skew(r_values),kurtosis(r_values),
                                                    np.mean(nir_values),np.var(nir_values),skew(nir_values),kurtosis(nir_values)]
    return statistics_df


def create_features(segments, B_band, G_band, R_band, NIR_band):
    """
    Create the features, based on the statistics values of the reflectances of each segment.
    
    Parameters
    ----------
    segments : array-like, shape (p,q)
        The map of the segments, where the value of each pixel is the segment it belongs to.
    B_band : array-like, shape (p,q)
        The blue band of the image.
    G_band : array-like, shape (p,q)
        The green band of the image.
    R_band : array-like, shape (p,q)
        The red band of the image.
    NIR_band : array-like, shape (p,q)
        The NIR band of the image.
    
    Returns
    -------
    pandas.DataFrame, shape (n,16)
        The DataFrame containing the 16 features for each of the n segments.
    """
    if (np.shape(NIR_band)!=np.shape(R_band)) or (np.shape(R_band)!=np.shape(G_band)) or (np.shape(G_band)!=np.shape(B_band)):
        raise Exception('The dimensions of the different bands are not equal.')
        
    Bband = np.array(B_band).flatten()
    Gband = np.array(G_band).flatten()
    Rband = np.array(R_band).flatten()
    NIRband = np.array(NIR_band).flatten()
    segments_ = np.array(segments).flatten()
    
    idx_sort = np.argsort(segments_)
    sorted_segments = segments_[idx_sort]
    _, idx_start, count = np.unique(sorted_segments, return_counts=True, return_index=True)
    res = np.split(idx_sort, idx_start[1:]) # splits the indices into separate arrays
    
    statistics_df=[]
    with warnings.catch_warnings(): 
        warnings.filterwarnings("ignore", message="Precision loss occurred in moment calculation due to catastrophic cancellation.")
        for list_indexes_superpixel in res:
            b_values=Bband[list_indexes_superpixel]
            g_values=Gband[list_indexes_superpixel]
            r_values=Rband[list_indexes_superpixel]
            nir_values=NIRband[list_indexes_superpixel]
            statistics_df.append([np.mean(b_values),np.var(b_values),skew(b_values),kurtosis(b_values),
                                  np.mean(g_values),np.var(g_values),skew(g_values),kurtosis(g_values),
                                  np.mean(r_values),np.var(r_values),skew(r_values),kurtosis(r_values),
                                  np.mean(nir_values),np.var(nir_values),skew(nir_values),kurtosis(nir_values)])
    statistics_df = pd.DataFrame(statistics_df)
    statistics_df.columns = ['mean_blue','var_blue','skew_blue','kurt_blue',
                             'mean_green','var_green','skew_green','kurt_green',
                             'mean_red','var_red','skew_red','kurt_red',
                             'mean_nir','var_nir','skew_nir','kurt_nir']
    
    return statistics_df


def add_spatial_feature(segments, boundaries, features, adjacent_segments):
    """
    Add the spatial features, i.e. the features of the neighboring segments.
    
    Parameters
    ----------
    segments : array-like, shape (p,q)
        The map of the segments, where the value of each pixel is the segment it belongs to.
    boundaries : array-like, shape (p,q)
        The boolean array where the pixels on the boundaries are marked as True, and the others as False.
    features : pandas.DataFrame, shape (n,16)
        The DataFrame containing the 16 features for each of the n segments.
    adjacent_segments : list of list, length n
        The list where the i-th element contains the adjacents segments of the i-th segment.
    
    Returns
    -------
    pandas.DataFrame, shape (n,32)
        The DataFrame containing the 32 features (16 features + 16 neighbors features) for each of the n segments.
    """

    neighbors_df = []
    features_array = np.array(features)
    for adjacent_seg in adjacent_segments:
        mean_adjacent = np.mean(features_array[adjacent_seg],axis=0)
        neighbors_df.append(mean_adjacent)
    neighbors_df = pd.DataFrame(neighbors_df)

    # we define the columns names for the df
    col_names=[]
    for t in features.columns:
        col_names.append(t+'_neighbors')

    neighbors_df.columns = col_names
    
    return(features.join(neighbors_df))

   
def add_features_from_cnn (y_segments_CNN, statistics_df_spatial):
    # for each segment, we add the label predicted by the cnn_rf
    df_cnn_rf = pd.DataFrame(y_segments_CNN)

    col_names=[]
    for t in df_cnn_rf.columns:
        col_names.append('cnn_pred_class_'+str(t))

    df_cnn_rf.columns = col_names

    statistics_df_with_cnn_own = statistics_df_spatial.join(df_cnn_rf)
    
    return(statistics_df_with_cnn_own)

def add_neighbors_features_from_cnn (y_segments_CNN, statistics_df_with_cnn_own, adjacent_segments):

    # for each segment, we compute the mean labels of the neighboring segments (cnn_rf map)

    neighbors_list_cnn = []

    for adjacent_seg in adjacent_segments:
        mean_adjacent = np.mean(np.array(y_segments_CNN)[adjacent_seg],axis=0)
        neighbors_list_cnn.append(mean_adjacent)

    neighbors_df_cnn = pd.DataFrame(neighbors_list_cnn)
    statistics_df_with_cnn_full = statistics_df_with_cnn_own.join(neighbors_df_cnn)
    
    return(statistics_df_with_cnn_full)


####################

def gb_predict(X_train, y_train, X_test, **gb_parameters):
    y_pred_gb=[]
    for i_class in range(np.shape(y_train)[1]):
        reg = GradientBoostingRegressor(**gb_parameters)
        reg.fit(X_train, y_train[:,i_class])
        y_pred_gb.append(reg.predict(X_test))
    y_pred_gb=np.transpose(y_pred_gb)
    return(y_pred_gb)

####################

def output_map_from_segments(segments, all_segments_pred):
    # returns the output map, from a given prediction on the segments
    all_segments_classes = np.argmax(all_segments_pred,axis=1)
    output_map = np.copy(segments)
    for index_segment in np.unique(segments):
        output_map[output_map==index_segment] = all_segments_classes[index_segment]
    return(output_map)

####################

def pixelwise_score(true_map,predicted_map, mask=None):
    # only compute the pixelwise score on the mask values
    if mask is None:
        score = np.sum(true_map==predicted_map)/(np.shape(true_map)[0]*np.shape(true_map)[1])
    else:
        score = np.sum(np.logical_and(mask,true_map==predicted_map))/np.sum(mask)
    return(score)
        
        
        
        
        
        