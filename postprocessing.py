# Standard libraries
import tensorflow as tf
import os
import random
import numpy as np
import argparse
import math
import json
import tensorflow_addons as tfa
from scipy.ndimage import label as label_image
import skimage
from skimage.transform import resize
import cv2

# MIScnn
from miscnn.neural_network.metrics import identify_axis
from miscnn.processing.data_augmentation import Data_Augmentation
from miscnn.processing.subfunctions.normalization import Normalization
from miscnn.processing.subfunctions.resize import Resize
from miscnn.processing.subfunctions.clipping import Clipping
from miscnn.processing.subfunctions.resampling import Resampling
from miscnn.processing.preprocessor import Preprocessor
from miscnn.neural_network.model import Neural_Network
from miscnn.neural_network.metrics import dice_soft, tversky_crossentropy
from miscnn.neural_network.architecture.unet.standard import Architecture
from miscnn.evaluation.cross_validation import run_fold, load_disk2fold
from miscnn.data_loading.interfaces.image_io import Image_interface
from miscnn.data_loading.data_io import Data_IO
from miscnn.data_loading.data_io import backup_evaluation

# Tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

# Either 'KVASIR' or 'INSTRUMENT'
DATASET = 'KVASIR'
# Softmax threshold
THRESHOLD = 0.7
# Minimum region size
MINIMUM = 0.01

#Generating interface: single channel with 2 classes (background and enhancing tumour)
interface = Image_interface(pattern = ".*"
                                 , img_type = 'rgb'
                                 , img_format = 'jpg'
                                 , classes = 2)
# path to data folder
data_path = os.path.join("/path/to/dataset", DATASET)
# Create the Data I/O object 
data_io = Data_IO(interface, data_path)

# Sample list
sample_list = data_io.get_indiceslist()
sample_list.sort()

# Data augmentation
data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=True, gamma=True, gaussian_noise=True)

# Data preprocessing parameters
sf_normalize = Normalization(mode='z-score')
sf_resize = Resize((512,512))
subfunctions = [sf_resize, sf_normalize]

# Create preprocessing class
pp = Preprocessor(data_io
                  , data_aug=None
                  , batch_size=1
                  , prepare_subfunctions=True
                  , subfunctions=subfunctions
                  , prepare_batches=False
                  , analysis="fullimage"
                  , use_multiprocessing=True)


def dice_soft(y_true, y_pred, smooth=0.00001):
    # Identify axis
    axis = identify_axis(y_true.get_shape())

    # Calculate required variables
    intersection = y_true * y_pred
    intersection = K.sum(intersection, axis=axis)
    y_true = K.sum(y_true, axis=axis)
    y_pred = K.sum(y_pred, axis=axis)

    # Calculate Soft Dice Similarity Coefficient
    dice = ((2 * intersection) + smooth) / (y_true + y_pred + smooth)

    # Obtain mean of Dice & return result score
    dice = K.mean(dice)
    return dice

def dice_soft_loss(y_true, y_pred):
    return 1-dice_soft(y_true, y_pred)

# Define architecture
architecture = Architecture()

# Create the Neural Network model
model = Neural_Network(architecture=architecture
                      , preprocessor=pp
                      , loss=dice_soft_loss
                      , metrics=[dice_soft]
                      , batch_queue_size=1
                      , workers=1
                      , learninig_rate=0.1
                      , loss_weights=None)


for sample_index in sample_list:
    a = np.load(os.path.join('path/to/evaluation',DATASET,'fold_0', sample_index + '.npy'))
    b = np.load(os.path.join('path/to/evaluation',DATASET,'fold_1', sample_index + '.npy'))
    c = np.load(os.path.join('path/to/evaluation',DATASET,'fold_2', sample_index + '.npy'))
    d = np.load(os.path.join('path/to/evaluation',DATASET,'fold_3', sample_index + '.npy'))
    e = np.load(os.path.join('path/to/evaluation',DATASET,'fold_4', sample_index + '.npy'))
    
    pred = np.mean([a,b,c,d,e],axis=0)
    pred = np.where(pred[:,:,1] > THRESHOLD, 1 , 0)
    pred = pred.astype('uint8')
    
    img = skimage.io.imread(os.path.join('path/to/image',DATASET,sample_index,'imaging.jpg'))
    pred = resize(pred,(img.shape[0], img.shape[1]),order=0, preserve_range=True)
    
    labelled_mask, num_labels = label_image(pred)
    minimum_cc_sum = int(MINIMUM * img.shape[0] * img.shape[1])
    
    for label in range(num_labels+1):
        if np.sum(pred[labelled_mask == label]) < minimum_cc_sum:
            pred[labelled_mask == label] = 0
    
    cv2.imwrite(os.path.join('path/to/output',DATASET,sample_index +'.png'),pred)