from coral_mapper_functions import *
import segments_regressor

import numpy as np
import pandas as pd
import pickle
from skimage.segmentation import felzenszwalb, slic, watershed, find_boundaries
from skimage.filters import sobel
from skimage.color import rgb2gray
import random
from sklearn.metrics import r2_score, accuracy_score
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import warnings

from collections import Counter


class coral_map():
    
    def __init__(self, img, segments=None, map_img=None, rf_pb=None, predicted_map_pb_rf=None, size_img=7):
        self.blue_band=img[0]
        self.green_band=img[1]
        self.red_band=img[2]
        self.nir_band=img[3]
        if len(img)==5:
            self.bathy = img[4]
        self.rgb_img=np.dstack((self.red_band,self.green_band,self.blue_band))
        self.segments=segments
        self.map_img=map_img
        if not map_img is None:
            self.nb_zones = np.max(map_img)+1
        else:
            self.nb_zones = None
        self.predicted_map_pb_rf=predicted_map_pb_rf
        self.rf_pb=rf_pb
        self.size_img = size_img
        
        
    def segmentation(self, nb_bands=3, method='fz', use_pb_pred=False, compact_pb_pred=False, split=False, *args, **kwargs):
        splitted_img = False
        if use_pb_pred:
            pb_pred_map = np.argmax(self.predicted_map_pb_rf,axis=2).astype(np.int16)
            if compact_pb_pred:
                img_for_segmentation = np.dstack([self.rgb_img, pb_pred_map]).astype(np.float16)
            else:
                img_for_segmentation = np.copy(self.rgb_img)
                for i in np.unique(pb_pred_map):
                    temp_array_class = np.zeros(np.shape(pb_pred_map))
                    temp_array_class[pb_pred_map==i] = 1
                    img_for_segmentation = np.dstack([self.rgb_img, temp_array_class])
            shape_x, shape_y = pb_pred_map.shape
            if (split) & ((shape_x>3000) or (shape_y>3000)):
                #split the img in 9
                splitted_img = True
                x_split = shape_x // 3
                y_split = shape_y // 3
                sub_images = []
                for i in range(3):
                    for j in range(3):
                        sub_image = img_for_segmentation[i*x_split:(i+1)*x_split, j*y_split:(j+1)*y_split]
                        sub_images.append(sub_image)
        elif nb_bands==4:
            img_for_segmentation = np.dstack((self.nir_band,self.red_band,self.green_band,self.blue_band))
        else:
            img_for_segmentation = np.copy(self.rgb_img)
        
        if splitted_img:
            segments = np.zeros((shape_x,shape_y))
            if (method=='felzeszwalb') or (method=='fz'):
                for i in range(3):
                    for j in range(3):
                        seg_part = felzenszwalb(sub_images[i*3+j], *args, **kwargs) 
                        segments[i*x_split:(i+1)*x_split, j*y_split:(j+1)*y_split] = seg_part + np.max(segments) + 1
            elif (method=='watershed') or (method=='ws'):
                for i in range(3):
                    for j in range(3):
                        gradient = sobel(rgb2gray(sub_images[i*3+j]))
                        segments[i*x_split:(i+1)*x_split, j*y_split:(j+1)*y_split] = watershed(gradient,*args,**kwargs) + np.max(segments)
            elif method=='slic':
                for i in range(3):
                    for j in range(3):
                        seg_part = slic(sub_images[i*3+j], *args, **kwargs) 
                        segments[i*x_split:(i+1)*x_split, j*y_split:(j+1)*y_split] = seg_part + np.max(segments)
            else:
                raise Exception("Segmentation method has not been recognized.")
            segments = segments.astype(np.int16)
        else:
            if (method=='felzenszwalb') or (method=='fz'):
                segments = felzenszwalb(img_for_segmentation,*args,**kwargs)
            elif (method=='watershed') or (method=='ws'):
                gradient = sobel(rgb2gray(img_for_segmentation))
                segments = watershed(gradient,*args,**kwargs)
            elif method=='slic':
                segments = slic(img_for_segmentation, *args,**kwargs)
            else:
                raise Exception("Segmentation method has not been recognized.")
        self.segments=segments
        # if the instance has some features, we delete it, because the segmentation has changed
        if hasattr(self, 'X_level_1__'):
            delattr(self, 'X_level_1__')
        if hasattr(self, 'smoothed_map_pb_rf'):
            delattr(self, 'smoothed_map_pb_rf')
        
        
    def create_augmented_images__(self):
        self.blue_augmented = create_augmented_image(self.blue_band)
        self.green_augmented = create_augmented_image(self.green_band)
        self.red_augmented = create_augmented_image(self.red_band)
        self.nir_augmented = create_augmented_image(self.nir_band)
        if hasattr(self, 'bathy'):
            self.bathy_augmented = create_augmented_image(self.bathy)
    
    def predict_pixelbased_rf(self, indexes, verbose=1, size_img=None):
        if size_img is None:
            size_img = self.size_img
        if hasattr(self, 'bathy'):
            superpixels = return_superpixels_features(indexes, self.blue_augmented, self.green_augmented, self.red_augmented, self.nir_augmented, self.bathy_augmented, size_img=size_img)
        else:
            superpixels = return_superpixels_features(indexes, self.blue_augmented, self.green_augmented, self.red_augmented, self.nir_augmented, size_img=size_img)
        superpixels=np.array(superpixels)
        y_pred_validation = self.rf_pb.predict(superpixels.reshape((superpixels.shape[0],np.prod(superpixels.shape[1:]))))
        return(y_pred_validation)
        
    def score_pixelbased_rf(self, indexes, y_true=None, verbose=1, size_img=None):
        if size_img is None:
            size_img = self.size_img
        y_pred=self.predict_pixelbased_rf(indexes, verbose=verbose, size_img=size_img)
        if y_true is None:
            y_true=return_labels(indexes, self.map_img)
        acc = sum(y_pred==y_true)/len(y_pred)
        return(acc)
    
    def train_pixelbased_rf(self, training_set_indexes, validation_set_indexes=[], undersampling=0, verbose=1, return_val_acc=False, *args, **kwargs):
        
        size_img = self.size_img
        
        labels = return_labels(training_set_indexes, self.map_img)
        if len(np.unique(self.map_img))!=len(np.unique(labels)):
            warnings.warn("The training set does not contain all the classes. Errors may occur.")
        
        if undersampling>0:
            if verbose:
                print("Undersampling...")
            indexes_to_delete = random_undersampling(labels,k=undersampling)
            for i in sorted(indexes_to_delete, reverse=True):
                del training_set_indexes[i]
            if verbose:
                print("Undersampling done. Number of points deleted:", len(indexes_to_delete))
                       
        if verbose:
            print("Creating the labels for each superpixel...")
        labels_training = return_labels(training_set_indexes, self.map_img)
        
        if not hasattr(self, 'blue_augmented'):
            self.create_augmented_images__()
                
        if verbose:
            print("Creating the features for each superpixel...")
        if hasattr(self, 'bathy'):
            superpixels = return_superpixels_features(training_set_indexes, self.blue_augmented, self.green_augmented, self.red_augmented, self.nir_augmented, self.bathy_augmented, size_img=size_img)
        else:
            superpixels = return_superpixels_features(training_set_indexes, self.blue_augmented, self.green_augmented, self.red_augmented, self.nir_augmented, size_img=size_img)
        
        superpixels=np.array(superpixels)
        
        if verbose:
            print("Training the Random Forest classifier...")
        X_train = superpixels.reshape((superpixels.shape[0],np.prod(superpixels.shape[1:])))
        self.rf_pb = RandomForestClassifier(*args, **kwargs)
        self.rf_pb.fit(X_train,labels_training)
        
        if len(validation_set_indexes)>0:
            if verbose:
                validation_acc = self.score_pixelbased_rf(validation_set_indexes, verbose=verbose, size_img=size_img)
                print("Validation accuracy:", validation_acc)
                if return_val_acc:
                    return(validation_acc)
            else:
                if return_val_acc:
                    validation_acc = self.score_pixelbased_rf(validation_set_indexes, verbose=verbose, size_img=size_img)
                    return(validation_acc)
         
    def load_rf_pixelbased(self, path='rf_pixelbased.pkl'):
        #self.rf_pb = pickle.load(open(path, 'rb'))
        with open(path, 'rb') as file:
            self.rf_pb = pickle.load(file)
            
    def load_regressor_round_2(self, path='rf_round_2.pkl'):
        self.rf_reg = pickle.load(open(path, 'rb'))
            
    def predict_map_pixelbased_rf(self, verbose=False, batch_length=10):
        
        if not hasattr(self, 'blue_augmented'):
            self.create_augmented_images__()
            
        self.predicted_map_pb_rf = np.empty((self.blue_band.shape[0],self.blue_band.shape[1],len(self.rf_pb.classes_)), dtype=np.float16)
        imin = 10-self.size_img//2
        imax = self.size_img+imin
        
        nb_rows=self.blue_band.shape[0]
        nb_col =self.blue_band.shape[1]
        if verbose:
            pbar = tqdm(total=nb_rows)
            
        for i in range(nb_rows//batch_length):            
            x_batch = []
            for k in range(batch_length):
                for j in range(nb_col):
                    index_i = i*batch_length+k
                    superpixel_blue = self.blue_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                    superpixel_green = self.green_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                    superpixel_red = self.red_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                    superpixel_nir = self.nir_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                    if hasattr(self, 'bathy'):
                        superpixel_bathy = self.bathy_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                        x_batch.append(np.dstack((superpixel_blue,superpixel_green,superpixel_red,superpixel_nir,superpixel_bathy)))
                    else:
                        x_batch.append(np.dstack((superpixel_blue,superpixel_green,superpixel_red,superpixel_nir)))
            x_batch = np.array(x_batch)
            batch_result = x_batch.reshape((x_batch.shape[0],np.prod(x_batch.shape[1:])))
            y_batch = self.rf_pb.predict_proba(batch_result)
            y_batch = y_batch.reshape((batch_length, y_batch.shape[0]//batch_length, y_batch.shape[1]))
            self.predicted_map_pb_rf[i*batch_length:(i+1)*batch_length,:,:] = y_batch
            if verbose:
                pbar.update(batch_length)
        
        remaining_rows = nb_rows%batch_length
        x_batch = []
        for k in range(remaining_rows):
            for j in range(self.blue_band.shape[1]):
                index_i = nb_rows - remaining_rows +k
                superpixel_blue = self.blue_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                superpixel_green = self.green_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                superpixel_red = self.red_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                superpixel_nir = self.nir_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                if hasattr(self, 'bathy'):
                    superpixel_bathy = self.bathy_augmented[index_i+imin:index_i+imax,j+imin:j+imax]
                    x_batch.append(np.dstack((superpixel_blue,superpixel_green,superpixel_red,superpixel_nir,superpixel_bathy)))
                else:
                    x_batch.append(np.dstack((superpixel_blue,superpixel_green,superpixel_red,superpixel_nir)))  
        if remaining_rows>0:
            x_batch = np.array(x_batch)
            batch_result = x_batch.reshape((x_batch.shape[0],np.prod(x_batch.shape[1:])))
            y_batch = self.rf_pb.predict_proba(batch_result)
            y_batch = y_batch.reshape((remaining_rows, y_batch.shape[0]//remaining_rows, y_batch.shape[1]))
            self.predicted_map_pb_rf[-remaining_rows:,:,:] = y_batch
                
        if verbose:
            pbar.close()    
            
    def smooth_pb_rf(self):
        if self.nb_zones is None:
            self.y_segments_pb_rf = create_labels_dataset(self.segments, np.argmax(self.predicted_map_pb_rf,axis=2))
        else:
            self.y_segments_pb_rf = create_labels_dataset(self.segments, np.argmax(self.predicted_map_pb_rf,axis=2), nb_zones=self.nb_zones)
        self.y_segments_pb_rf_classes = np.argmax(self.y_segments_pb_rf,axis=1)
        self.smoothed_map_pb_rf = create_map_from_segments(self.segments,self.y_segments_pb_rf_classes)
        
    def init_regressor_round_2(self, features_level=1):
        boundaries = find_boundaries(self.segments)
        adjacent_segments,_ = segments_regressor.find_adjacent_segments(self.segments, boundaries)
        
        if not hasattr(self, 'X_level_1__'):
            self.X_level_1__ = segments_regressor.create_features(self.segments, self.blue_band, self.green_band, self.red_band, self.nir_band)
        
        if features_level==1:
            self.X = self.X_level_1__
        elif features_level==2:
            self.X = segments_regressor.add_spatial_feature(self.segments, boundaries, self.X_level_1__, adjacent_segments)
        elif features_level==3:
            features_spatial = segments_regressor.add_spatial_feature(self.segments, boundaries, self.X_level_1__, adjacent_segments)
            self.X = segments_regressor.add_features_from_pb(self.y_segments_pb_rf, features_spatial)
        elif features_level==4:
            features_spatial = segments_regressor.add_spatial_feature(self.segments, boundaries, self.X_level_1__, adjacent_segments)
            features_with_pb_own = segments_regressor.add_features_from_pb(self.y_segments_pb_rf, features_spatial)
            self.X = segments_regressor.add_neighbors_features_from_pb(self.y_segments_pb_rf, features_with_pb_own, adjacent_segments)
        else:
            raise Exception('features_level should be between 1 and 4.')
        self.X = self.X.fillna(0)
        
    def train_regressor_round_2(self, train_indexes=None, test=True, return_metrics=False, verbose=True, adjust_segments_threshold=0, adjust_segments_class=0, random_seed=0, *args, **kwargs):
        
        self.rf_reg = RandomForestRegressor(**kwargs)
        self.y_true_segments = create_labels_dataset(self.segments,self.map_img)
        
        if adjust_segments_threshold>0:
            cond = self.y_true_segments[:,adjust_segments_class] < adjust_segments_threshold
            y_true_segments_final = self.y_true_segments[cond]
            y_true_segments_final = np.delete(y_true_segments_final, adjust_segments_class, 1)
            y_true_segments_final = y_true_segments_final / y_true_segments_final.sum(axis=1)[:, np.newaxis]
            X_final = self.X[cond]
        else:
            y_true_segments_final = self.y_true_segments
            X_final = self.X
            
        if train_indexes is None:
            sample_indices = np.arange(len(y_true_segments_final))
            random.seed(random_seed)
            size_train = 4*len(sample_indices)//5
            train_indexes = np.random.choice(sample_indices,size=size_train,replace=False) 
        
        self.rf_reg.fit(np.array(X_final)[train_indexes], y_true_segments_final[train_indexes])
        if test:
            y_pred_test = self.rf_reg.predict(np.delete(np.array(X_final),train_indexes,axis=0))
            y_true_test = np.delete(y_true_segments_final,train_indexes,axis=0)
            r2_score_test = r2_score(y_true_test, y_pred_test)
            acc_test = accuracy_score(np.argmax(y_true_test, axis=1), np.argmax(y_pred_test, axis=1))
            if verbose:
                print('R2 score (on test set): ', r2_score_test)
                print('Class accuracy (on test set): ', acc_test)
        if adjust_segments_threshold > 0:
            if return_metrics:
                return(train_indexes, cond, r2_score_test, acc_test)
            else:
                return(train_indexes, cond)
        else:
            if return_metrics:
                return(train_indexes, r2_score_test, acc_test)
            else:
                return(train_indexes)
            
    def predict_map_round_2(self):
        y_pred_classes = np.argmax(self.rf_reg.predict(np.array(self.X)), axis=1)
        self.predicted_map_round_2 = create_map_from_segments(self.segments,y_pred_classes)
        
    def predict_map_round_2_with_object_based_rf(self):
        y_pred = self.rf_reg.predict(np.array(self.X))
        created_map = np.zeros((np.shape(self.segments)[0],np.shape(self.segments)[1],np.shape(y_pred)[-1]))
        for index_segment in np.unique(self.segments):
            created_map[self.segments==index_segment] = y_pred[index_segment]
        self.predicted_map_round_2_with_object_based_rf = created_map
        
    def post_processing(self, connectivity, threshold_w=0.8, threshold_a=1.):
        """
        Parameters:
        -----------
        connectivity : pandas.DataFrame
        threshold_w : float, default=0.8
            The threshold for the segments that are considered wrong according to the connectivity table.
        threshold_a : float, default=0.9
            The threshold for all the other segments.
        """
        if threshold_a < threshold_w:
            raise Exception("The threshold must be lower for the wrong segments than for the others.")
        
        segments_classes = np.copy(self.y_segments_pb_rf_classes)
        boundaries = find_boundaries(self.segments)
        adjacent_segments,_ = segments_regressor.find_adjacent_segments(self.segments, boundaries)
        
        list_wrong_neighbours = []
        for i in range(len(adjacent_segments)):
            wrong_neighbours = 0
            s = adjacent_segments[i]
            s_class = segments_classes[i]
            if s_class > 0:
                s_connectivity = connectivity[s_class]
                for j in s:
                    if segments_classes[j]>0:
                        if s_connectivity[segments_classes[j]] == 0 :
                            wrong_neighbours += 1
            list_wrong_neighbours.append(wrong_neighbours/len(s))
            
        for i, value in enumerate(list_wrong_neighbours):
            lst_temp = [segments_classes[j] for j in adjacent_segments[i]]
            majority_neighbours = max(set(lst_temp), key=lst_temp.count)
            if value>0:
                if majority_neighbours/len(lst_temp) > threshold_w:
                    segments_classes[i] = majority_neighbours
            else:
                if majority_neighbours/len(lst_temp) > threshold_a:
                    segments_classes[i] = majority_neighbours
        
        #recreate the map
        post_processed_map = np.copy(self.segments)
        for i in range(np.max(self.segments)+1):
            post_processed_map[self.segments==i] = segments_classes[i]+1
        self.post_processed_map = post_processed_map