"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from utils import *
import torch.nn as nn

class SplitDataSet(object):
    """Dataset class object."""
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=False,
                 patch_length = 128,
                 add_axis = False,
                 normalize = False):
        """Initialize the class."""
#         print(images.shape)
#         print(images[0])
#         if reshape:            
#             images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])               
        if normalize:
            images = _normalize_img(images)
        if add_axis:
            images = _add_axis(images) 
        self._images = images    
        self._num_examples = images.shape[0]
        self._labels = np.array(labels)
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._bag_lists, self._ins_per_bag = self._creat_bags(patch_length)
        self._bag_epochs_completed = 0
        self._bag_index_in_epoch = 0       
        if one_hot:
            self._one_hot_labels = _to_one_hot(self.labels,2)

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels
    
    @property
    def bags(self):
        return self._bag_lists

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    @property
    def bag_epochs_completed(self):
        return self._bag_epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            #self._labels = np.array(self._labels)
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]
    
    def next_bag_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        start = self._bag_index_in_epoch
        self._bag_index_in_epoch += batch_size
        if self._bag_index_in_epoch > self._num_examples:
            # Finished epoch
            self._bag_epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._bag_lists = self._bag_lists[perm]
            #self._labels = np.array(self._labels)
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._bag_index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._bag_index_in_epoch
        return self._bag_lists[start:end], self._labels[start:end]
    
    def _creat_bags(self,patch_length):
        """transform all frames into bags of instances with one label"""
        '''return : bags[num_imgs, num_instance_in_bag, patch_length, patch_length], labels[num_imgs,one_hot_bag_label]'''
        #print(self.images.shape)
        bags_list = []
        #bags_list = np.array([])
        _,instance_per_bag = self._frame_to_bag(self.images[0],patch_length)
        for i in range(self._num_examples):
            one_bag,_ = self._frame_to_bag(self.images[i],patch_length)
            bags_list.append(one_bag) 
            #bags_list = np.append(bags_list,one_bag,axis=0)
        bags_list = np.asarray(bags_list)
        #print("bag_list shape",bags_list.shape)
        return bags_list,instance_per_bag
    
    def _frame_to_bag(self, images, patch_length):
        #print(images.shape)
        """one frame = one bag = multiple instance, instance = patch_length * patch_length"""
        n_split = 2*int(images.shape[0]/patch_length) -1
        n_patch = n_split**2
        patch_list = []
        #patch_list = np.array([])
        l = 0
        u = 0
        while l < images.shape[0] - patch_length/2 :
            u = 0
            while u < images.shape[0] - patch_length/2 :
               # print(l, u)
                p = [images[l:l+patch_length,u:u+patch_length]]
                #print("patch length",len(p))
                #p = np.asarray(p)
                #print("patch shape",p.shape)
                patch_list.append(p)
                #patch_list = np.append(patch_list,p,axis=0)
                u+=int(patch_length/2)
            l+=int(patch_length/2)        
        #print("patch_list len",len(patch_list))
        return patch_list,n_patch

    def _normalize_img(img):
        img = img - np.min(img); 
        img = img/(np.max(img)+0.0001)
        return img
    
    def _add_axis(img):
        return img[:,:,:,np.newaxis]

    def _to_one_hot(labels,num_classes):
        """Convert '0' to [1,0], convert '1' to [0,1] """
        one_hot_labels = []
        for label in labels: 
            if label == 0:
                one_hot=np.array([1.,0.])
            else:
                one_hot=np.array([0.,1.])
            one_hot_labels.append(one_hot)
        one_hot_labels=np.array(one_hot_labels)
        return one_hot_labels
    



def to_scalar_label(areas,num_classes):
    """Convert '60-' to 0, convert '60+'to 1 """
    scalar_labels = []
    for a in areas: 
        if a < 60:
            label=0
        else:
            label=1
        scalar_labels.append(label)
    scalar_labels=np.array(scalar_labels)
    return scalar_labels

def normalize_img(img):
    img = img - np.min(img); 
    img = img/(np.max(img)+0.0001)
    return img

def read_data_sets(folder = '/media/data/sifan/Documents/data/', fake_data=False, one_hot=True,
                        dtype=dtypes.float64, reshape=True):
    """Set the images and labels."""
    train = np.load(folder + 'train_indx.npy').item()
    val = np.load(folder + 'val_indx.npy').item()
    test = np.load(folder + 'test_indx.npy').item()
    
    train_images = normalize_img(train['imgs'])
    train_area = train['area']
    val_images = normalize_img(val['imgs'])
    val_area = val['area']
    test_images = normalize_img(test['imgs'])
    test_area = test['area']       
    train_labels = to_scalar_label(train_area,2)
    val_labels = to_scalar_label(val_area,2)
    test_labels = to_scalar_label(test_area,2)
        
    train = SplitDataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    val = SplitDataSet(val_images, val_labels, dtype=dtype,reshape=reshape)
    test = SplitDataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
    ds = collections.namedtuple('SplitDataSet', ['train', 'val', 'test'])

    return ds(train=train, val=val, test=test)

class AngioBags(data_utils.Dataset):
    def __init__(self, seed=123,_is_train=False,_is_val=False,_is_test=False):
        self.data = read_data_sets(one_hot=True)
        
        self._is_train = _is_train
        self._is_val = _is_val
        self._is_test = _is_test
        if _is_train:
            self.train =  self.data.train
            self.nTrain = self.data.train.num_examples
            self.train_bags =  self.data.train._bag_lists
            self.train_labels = self.data.train._labels
            print("train bag shape",self.train_bags.shape)
            
        elif _is_val:
            self.val =  self.data.val
            self.nVal = self.data.val.num_examples
            self.val_bags =  self.data.val._bag_lists
            self.val_labels = self.data.val._labels
            print("val bag shape",self.val_bags.shape)
            
        else:
            self.test =  self.data.test     
            self.nTest = self.data.test.num_examples
            self.test_bags =  self.data.test._bag_lists
            self.test_labels = self.data.test._labels 
            print("test bag shape",self.test_bags.shape)
                    
        self.r = np.random.RandomState(seed)


    def __len__(self):
        if self._is_train:
            return self.nTrain
        elif self._is_val:
            return self.nVal
        else:
            return self.nTest

    def __getitem__(self, index):
        if self._is_train:
            bag = self.train_bags[index]
            label = self.train_labels[index]
        elif self._is_val:
            bag = self.val_bags[index]
            label = self.val_labels[index]
        else:
            bag = self.test_bags[index]
            label = self.test_labels[index]

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(AngioBags(seed=123,
                                                   _is_train=True),
                                         batch_size=1,
                                         shuffle=True)
    
    val_loader = data_utils.DataLoader(AngioBags(seed=123,
                                                 _is_val=True),
                                       batch_size=1,
                                       shuffle=True)
    
    test_loader = data_utils.DataLoader(AngioBags(seed=123,
                                                  _is_test=True),
                                        batch_size=1,
                                        shuffle=True)

    angio_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        angio_bags_train += label[0]
    print('Positive Number in train bags: {}/{}\n'
          'Number of instances per bag, {} \n'.format(
        angio_bags_train, len(train_loader),train_loader.dataset.train._ins_per_bag))
    
    angio_bags_val = 0
    for batch_idx, (bag, label) in enumerate(val_loader):
        angio_bags_val += label[0]
    print('Positive Number in valid bags: {}/{}\n'
          'Number of instances per bag, {} \n'.format(
        angio_bags_val, len(val_loader),val_loader.dataset.val._ins_per_bag))
    
    angio_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        angio_bags_test += label[0]
    print('Positive Number in test bags: {}/{}\n'
          'Number of instances per bag, {} \n'.format(
        angio_bags_test, len(test_loader),test_loader.dataset.test._ins_per_bag))
    
