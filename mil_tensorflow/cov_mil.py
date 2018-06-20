import collections
from tensorflow.python.framework import dtypes
from TensorBase.tensorbase.base import Model
from TensorBase.tensorbase.base import Layers
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

class MDataSet(object):
    """Dataset class object."""

    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=True):
        """Initialize the class."""
        if reshape:
            
            print(images.shape)
            images.reshape(images.shape[0],images.shape[1] * images.shape[2])                

        self._images = images
        self._num_examples = images.shape[0]
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

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
            #print(self._images.shape)
            #print(self._labels.shape)
            self._images = self._images[perm]
            self._labels = np.array(self._labels)
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        #print(start)
        return self._images[start:end], self._labels[start:end]


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


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

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

class Conv_MIL(Model):
    def __init__(self, flags_input):
        """ Initialize from Model class in TensorBase """
        
        self.flags = flags_input
        self.flags['DIR_PATH'] = '/media/data/sifan/MIL/mil_tensorflow/'
        self.flags['save_directory']=self.flags['DIR_PATH'] + 'conv_mil/'
        self.flags['model_directory'] = self.flags['DIR_PATH'] + 'summaries/' 
        self.flags['seed'] = 123
        self.flags['restore_directory']=self.flags['save_directory']
        
        super().__init__(self.flags)
        self.checkpoint_rate = 2  # save after this many epochs
        self.valid_rate = 2  # validate after this many epochs
        
        self.valid_results = []
        self.test_results = []
        self.data = read_data_sets(one_hot=True)
        self.path = '/media/data/sifan/Documents/multi_instance_learning'
        self.patch_length = self.flags['PATCH_LENGTH']       
        self.nTrain = self.data.train.num_examples
        self.nVal = self.data.validation.num_examples
        self.nTest = self.data.test.num_examples

    def _data(self):
        """ Define all data-related parameters. Called by TensorBase. """
        self.data = read_data_sets(one_hot=True)
        self.path = self.flags['DIR_PATH']
        self.patch_length = self.flags['PATCH_LENGTH']      
        self.nTrain = self.data.train.num_examples
        self.nVal = self.data.validation.num_examples
        self.nTest = self.data.test.num_examples
        self.x = tf.placeholder(tf.float32, [None, 512, 512, 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[None,self.flags['NUM_CLASSES']], name='y')

    def _summaries(self):
        """ Write summaries out to TensorBoard. Called by TensorBase. """
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("XEntropy_Loss_Pi", self.xentropy_p)
        tf.summary.scalar("XEntropy Loss_yi", self.xentropy_y)
        tf.summary.scalar("Weight_Decay_Loss", self.weight)
        

    def print_summaries(self):
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)
            print(var.name)
        print("Total_Loss", self.cost)
        print("XEntropy_Loss_Pi", self.xentropy_p)
        print("XEntropy Loss_yi", self.xentropy_y)
        print("Weight_Decay_Loss", self.weight)
        
    def _network(self):
        """ Define neural network. Uses Layers class of TensorBase. Called by TensorBase. """
        with tf.variable_scope("model"):
            net = Layers(self.x)
            net.conv2d(5, 64)
            net.maxpool()
            net.conv2d(3, 64)
            net.conv2d(3, 64)
            net.maxpool()
            net.conv2d(3, 128)
            net.conv2d(3, 128)
            net.maxpool()
            net.conv2d(1, 2, activation_fn=tf.nn.sigmoid)
            net.noisy_and(2)
            self.P_i = net.get_output()
            net.fc(2)
            self.y_hat = net.get_output()
            self.logits = tf.nn.softmax(self.y_hat)

    def _optimizer(self):
        """ Set up loss functions and choose optimizer. Called by TensorBase. """
        const = 1/self.flags['BATCH_SIZE']
        self.xentropy_p = const * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.P_i, name='xentropy_p'))
        self.xentropy_y = const * tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat, name='xentropy_y'))
        #logits = tf.reduce_mean(self.y_hat,0)
        self.weight = self.flags['WEIGHT_DECAY'] * tf.add_n(tf.get_collection('weight_losses'))
        self.cost = tf.reduce_sum(self.xentropy_p + self.xentropy_y + self.weight)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.flags['LEARNING_RATE']).minimize(self.cost)

    def train(self):
        """ Run training function for num_epochs. Save model upon completion. """
        print('Training for %d epochs' % self.flags['NUM_EPOCHS'])

        for self.epoch in range(1, self.flags['NUM_EPOCHS'] + 1):
            for _ in tqdm(range(int(float(self.nTrain)))):

                # Get minibatches of data
                batch_x, batch_y = self.data.train.next_batch(self.flags['BATCH_SIZE'])
                #print("batch_x:",batch_x)
#                 print("batch_y_shape:",batch_y.shape)
#                 print("self.P_i_shape:",self.P_i.shape)
#                 print("self.P_i:",self.P_i)
#                 print("self.y_hat_shape:",self.y_hat.shape)
#                 print("self.y_hat:",self.y_hat)
                batch_x = self.reshape_batch(batch_x)

                # Run a training iteration
                summary, loss, _ , _= self.sess.run([self.merged, self.cost, self.optimizer,self.weight],
                                                   feed_dict={self.x: batch_x, self.y: batch_y})
                #print("loss:" ,loss)
                self._record_training_step(summary)
            #self.print_summaries()
            
            if self.step % (self.flags['DISPLAY_STEP']) == 0:
                # Record training metrics every display_step interval
                self._record_train_metrics(loss)

            ## Epoch finished
            # Save model
            if self.epoch % self.checkpoint_rate == 0:
                self._save_model(section=self.epoch)
            # Perform validation
            if self.epoch % self.valid_rate == 0:
                self.evaluate('valid')

    def evaluate(self, dataset):
        """ Evaluate network on the valid/test set. """

        # Initialize to correct dataset
        print('Evaluating images in %s set' % dataset)
        print(self.weight)
        if dataset == 'valid':
            num_images = self.nVal
            results = self.valid_results
            for _ in tqdm(range(int(float(num_images)))):
                batch_x, batch_y = self.data.validation.next_batch(self.flags['BATCH_SIZE'])
                batch_x = self.reshape_batch(batch_x)
                feed_dict={self.x: batch_x}
                logits,_ = self.sess.run([self.logits,self.weight], feed_dict={self.x: batch_x})
#                 with self.sess.as_default():
#                     logits = self.logits.eval(feed_dict=feed_dict)
                #print("logits:",logits)
                #print("batch_y:",batch_y)
                predictions = np.reshape(logits, [-1, self.flags['NUM_CLASSES']])
                #predictions = np.mean(logits[0], axis=0)
                #print("predictions:",predictions)
#                 if predictions[0]>predictions[1]:
#                     predictions = np.asarray([1.,0.])
#                 else:
#                     predictions = np.asarray([0.,1.]) 
                correct_prediction = np.equal(np.argmax(batch_y, 1), np.argmax(predictions, 1))
                #correct_prediction = np.equal(batch_y, predictions)
                acc = float(correct_prediction.sum())/float(len(correct_prediction))
                #print("correct_prediction:",correct_prediction)
                #print("val_acc:",acc,end = '\r',flush = True)
                #results = np.concatenate((self.valid_results, acc))
                self.valid_results.append(acc)


        else:
#             batch_x, batch_y = self.mnist.test.next_batch(self.flags['BATCH_SIZE'])
#             batch_x = self.reshape_batch(batch_x)
            num_images = self.nTest
            results= self.test_results
            for _ in tqdm(range(int(float(num_images)))):
                batch_x, batch_y = self.data.test.next_batch(self.flags['BATCH_SIZE'])
                batch_x = self.reshape_batch(batch_x)
                logits = self.sess.run([self.logits], feed_dict={self.x: batch_x})
                #print("batch_y:",batch_y)
                #print("logits:",logits)
                predictions = np.reshape(logits, [-1, self.flags['NUM_CLASSES']])
                #predictions = np.mean(logits[0], axis=0)
                #print("predictions:",predictions)
#                 if predictions[0]>predictions[1]:
#                     predictions = np.asarray([1.,0.])
#                 else:
#                     predictions = np.asarray([0.,1.])
                correct_prediction = np.equal(np.argmax(batch_y, 1), np.argmax(predictions, 1))
                #correct_prediction = np.equal(batch_y, predictions)
                acc = float(correct_prediction.sum())/float(len(correct_prediction))
                #print("correct_prediction:",correct_prediction,end='\r',flush=True)
                #print("test_acc:",acc,end='\r',flush=True)
                self.test_results.append(acc)
                #results = np.concatenate((self.valid_results, correct_prediction))


#         print("batch_x:",batch_x.shape)
#         print("batch_y:",batch_y.shape)
#         print("results:",results.shape)
        
        # Loop through all images in eval dataset
        
        # Calculate average accuracy and record in text file
        self._record_eval_metrics(dataset)

        # Calculate average accuracy and record in text file
        #self._record_eval_metrics(dataset)

    #########################
    ##   Helper Functions  ##
    #########################

    def reshape_batch(self, batch):
        """ Reshape vector into image. Do not need if data that is loaded in is already in image-shape"""
        return np.reshape(batch, [-1, 512, 512, 1])
        #return np.reshape(batch, [self.flags['BATCH_SIZE'], self.flags['PATCH_LENGTH'], self.flags['PATCH_LENGTH'], 1])

    def _record_train_metrics(self, loss):
        """ Records the metrics at every display_step iteration """
        print("Batch Number: " + str(self.step) + ", Total Loss= " + "{:.6f}".format(loss))

    def _record_eval_metrics(self, dataset):
        """ Record the accuracy on the eval dataset """
        if dataset == 'valid':
            accuracy = np.mean(self.valid_results)
        else:
            accuracy = np.mean(self.test_results)
        print("Accuracy on %s Set: %f" % (dataset, float(accuracy)))
        file = open(self.flags['restore_directory'] + dataset + 'Accuracy.txt', 'w')
        file.write('%s set accuracy:' % dataset)
        file.write(str(accuracy))
        file.close()
        
def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Faster R-CNN Networks Arguments')
    parser.add_argument('-n', '--RUN_NUM', default=0,type=int)  # Saves all under /save_directory/model_directory/Model[n]
    parser.add_argument('-e', '--NUM_EPOCHS', default=50,type=int)  # Number of epochs for which to train the model
    parser.add_argument('-r', '--RESTORE_META', default=0)  # Binary to restore from a model. 0 = No restore.
    parser.add_argument('-m', '--MODEL_RESTORE', default=0)  # Restores from /save_directory/model_directory/Model[n]
    parser.add_argument('-f', '--FILE_EPOCH', default=1)  # Restore filename: 'part_[f].ckpt.meta'
    parser.add_argument('-t', '--TRAIN', default=1)  # Binary to train model. 0 = No train.
    parser.add_argument('-v', '--EVAL', default=1)  # Binary to evaluate model. 0 = No eval.
    parser.add_argument('-l', '--LEARNING_RATE', default=1e-3, type=float)  # learning Rate
    parser.add_argument('-g', '--GPU', default=1,type=int)  # specify which GPU to use
    parser.add_argument('-s', '--SEED', default=77,type=int)  # specify the seed
    parser.add_argument('-d', '--MODEL_DIRECTORY', default='summaries/', type=str)  # To save all models
    parser.add_argument('-a', '--SAVE_DIRECTORY', default='/media/data/sifan/Documents/multi_instance_learning/conv_mil/', type=str)  # To save individual run
    parser.add_argument('-i', '--DISPLAY_STEP', default=1, type=int)  # how often to display metrics
    parser.add_argument('-b', '--BATCH_SIZE', default=1, type=int)  # size of minibatch
    parser.add_argument('-w', '--WEIGHT_DECAY', default=1e-3, type=float)  # decay on all Weight variables
    parser.add_argument('-c', '--NUM_CLASSES', default=2, type=int)  # number of classes. proly hard code.
    parser.add_argument('-p', '--PATCH_LENGTH', default=512, type=int)  # length of patches. proly hard code.
    flags = vars(parser.parse_args())

    # Run model. Train and/or Eval.
    model = Conv_MIL(flags)
    #tf.global_variables_initializer().run()
    if int(flags['TRAIN']) == 1:
        model.train()
    if int(flags['EVAL']) == 1:
        model.evaluate('test')

if __name__ == "__main__":
    main()
       
