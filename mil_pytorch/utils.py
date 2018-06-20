import collections
from tensorflow.python.framework import dtypes
# from TensorBase.tensorbase.base import Model
# from TensorBase.tensorbase.base import Layers
import sys
sys.path.append('../')
import tensorflow as tf
import numpy as np
import argparse
from tqdm import tqdm
import numpy as np
import pdb
from scipy.ndimage.filters import gaussian_filter
import cv2
import pywt
from utils import *
import pydicom
import pickle

def read_data_sets(folder = '/media/data/sifan/Documents/data/', fake_data=False, one_hot=True,
                        dtype=dtypes.float64, reshape=True):
    """Set the images and labels."""
    #train = pickle.load(open(folder + 'train_indx.npy',"rb"))
    train = np.load(folder + 'train_indx.npy').item()
    #val = pickle.load(open(folder + 'val_indx.npy',"rb"))
    val = np.load(folder + 'val_indx.npy').item()
    #test = pickle.load(open(folder + 'test_indx.npy',"rb"))
    test = np.load(folder + 'test_indx.npy').item()
    
    train_images = normalize_img(train['imgs'])
    train_images = train_images[:,:,:,np.newaxis]
    train_area = train['area']
    val_images = normalize_img(val['imgs'])
    val_images = val_images[:,:,:,np.newaxis]
    val_area = val['area']
    test_images = normalize_img(test['imgs'])
    test_images = test_images[:,:,:,np.newaxis]
    test_area = test['area']
    
    print("train_images:",train_images.shape)
    print("val_images:",val_images.shape)
    print("test_images:",test_images.shape)
    
    if one_hot is True:
        train_labels = _to_one_hot(train_area,2)
        val_labels = _to_one_hot(val_area,2)
        test_labels = _to_one_hot(test_area,2)

    train = MDataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = MDataSet(val_images, val_labels, dtype=dtype,
        reshape=reshape)

    test = MDataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    ds = collections.namedtuple('MDatasets', ['train', 'validation', 'test'])

    return ds(train=train, validation=validation, test=test)

def _to_one_hot(areas,num_classes):
    """Convert '60-' to [1,0], convert '60+' to [0,1] """
    one_hot_labels = []
    for a in areas: 
        #print("area:",a)
        if a < 60:
            one_hot=np.array([1.,0.])
        else:
            one_hot=np.array([0.,1.])
        #print(one_hot)
        one_hot_labels.append(one_hot)
        #one_hot_labels = np.concatenate(one_hot_labels,one_hot)
    one_hot_labels=np.array(one_hot_labels)
    print(one_hot_labels.shape)
    return one_hot_labels

def normalize_img(img):
    # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    # img = clahe.apply(img)
    img = img - np.min(img); 
    img = img/(np.max(img)+0.0001)
    return img

# def next_bag_batch(self, batch_size, fake_data=False):
#         """Return the next `batch_size` examples from this data set."""
#         start = self._bag_index_in_epoch
#         self._bag_index_in_epoch += batch_size
#         if self._bag_index_in_epoch > self._num_examples:
#             # Finished epoch
#             self._bag_epochs_completed += 1
#             # Shuffle the data
#             perm = np.arange(self._num_examples)
#             np.random.shuffle(perm)
#             self._bag_lists = self._bag_lists[perm]
#             #self._labels = np.array(self._labels)
#             self._labels = self._labels[perm]
#             # Start next epoch
#             start = 0
#             self._bag_index_in_epoch = batch_size
#             assert batch_size <= self._num_examples
#         end = self._bag_index_in_epoch
#         return self._bag_lists[start:end], self._labels[start:end]
    
