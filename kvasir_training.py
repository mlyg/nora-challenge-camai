# Standard libraries
import tensorflow as tf
import os
import random
import numpy as np
import argparse
import math

# MIScnn
from miscnn.neural_network.metrics import identify_axis
from miscnn.processing.data_augmentation import Data_Augmentation
from miscnn.processing.subfunctions.normalization import Normalization
from miscnn.processing.subfunctions.resize import Resize
from miscnn.processing.subfunctions.clipping import Clipping
from miscnn.processing.subfunctions.resampling import Resampling
from miscnn.processing.preprocessor import Preprocessor
from miscnn.neural_network.model import Neural_Network
from miscnn.neural_network.metrics import dice_soft

from miscnn.neural_network.architecture.unet.model import Architecture 

from miscnn.evaluation.cross_validation import run_fold, load_disk2fold
from miscnn.data_loading.interfaces.image_io import Image_interface
from miscnn.data_loading.data_io import Data_IO

# Tensorflow
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping

# Experiment seed
seed_value= 999
# Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)
# Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# Set `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)


# Argument is fold to specify which Cross validation fold to train
parser = argparse.ArgumentParser(description="KVASIR fold")
parser.add_argument("-f", "--fold", help="",
                    required=True, type=int, dest="fold")
args = parser.parse_args()
fold = args.fold


interface = Image_interface(pattern = "img[0-9]*"
                                 , img_type = 'rgb'
                                 , img_format = 'jpg'
                                 , classes = 2)
# path to data folder
data_path ="data path here"

# Generating dataloader
data_io = Data_IO(interface, data_path, delete_batchDir=False)

# Sample list
sample_list = data_io.get_indiceslist()
sample_list.sort()

# Basic checks
print("All samples: " + str(len(sample_list)))
sample = data_io.sample_loader(sample_list[0], load_seg=True)  
print("Image dimension check:",sample.img_data.shape, sample.seg_data.shape)

# Data augmentation
data_aug = Data_Augmentation(cycles=1, scaling=True, rotations=True, elastic_deform=True, mirror=True,
                             brightness=True, contrast=False, gamma=False, gaussian_noise=False)

# Data preprocessing parameters
sf_normalize = Normalization(mode='z-score')
sf_resize = Resize((512,512))
subfunctions = [sf_resize, sf_normalize]

# Create preprocessing class
pp = Preprocessor(data_io
                  , data_aug=data_aug
                  , batch_size=1
                  , prepare_subfunctions=True
                  , subfunctions=subfunctions
                  , prepare_batches=False
                  , analysis="fullimage"
                  , use_multiprocessing=False)


def asymmetric_focal_loss(delta=0.5, gamma=1.,boundary=True):
    def loss_function(y_true, y_pred):

        axis = identify_axis(y_true.get_shape())  

        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred)

        back_ce = K.pow(1 - y_pred[:,:,:,0], gamma) * cross_entropy[:,:,:,0]
        back_ce =  (1 - delta) * back_ce

        fore_ce = cross_entropy[:,:,:,1]
        fore_ce = delta * fore_ce

        loss = K.mean(K.sum(tf.stack([back_ce, fore_ce],axis=-1),axis=-1))

        return loss

    return loss_function

def asymmetric_focal_tversky_loss(delta=0.5, gamma=1):
    def loss_function(y_true, y_pred):
        axis = identify_axis(y_true.get_shape())
        
        # Clip values to prevent division by zero error
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate true positives (tp), false negatives (fn) and false positives (fp)     
        tp = K.sum(y_true * y_pred, axis=axis)
        fn = K.sum(y_true * (1-y_pred), axis=axis)
        fp = K.sum((1-y_true) * y_pred, axis=axis)
        dice_class = (tp + epsilon)/(tp + delta*fn + (1-delta)*fp + epsilon)

        #calculate losses separately for each class, only suppressing background class
        back_dice = (1-dice_class[:,0])
        fore_dice = (1-dice_class[:,1]) * K.pow(1-dice_class[:,1], -gamma) 

        # Sum up classes to one score
        loss = K.mean(tf.stack([back_dice,fore_dice],axis=-1))

        return loss

    return loss_function

def unified_focal_loss(delta=0.6, gamma=0.1):
    def loss_function(y_true,y_pred):
        # Obtain Focal Dice loss
        asymmetric_ftl = asymmetric_focal_tversky_loss(delta=delta, gamma=gamma)(y_true,y_pred)
        # Obtain Focal loss
        asymmetric_fl = asymmetric_focal_loss(delta=delta, gamma=gamma)(y_true,y_pred)

        return asymmetric_ftl + asymmetric_fl

    return loss_function


architecture = Architecture()


loss = unified_focal_loss(delta=0.6, gamma=0.3)


# Create the Neural Network model
model = Neural_Network(architecture=architecture
                      , preprocessor=pp
                      , loss=loss
                      , metrics=[dice_soft]
                      , batch_queue_size=3
                      , workers=1
                      , learninig_rate=1e-3
                      , loss_weights = None)


fold_path = "fold path here"

training, validation = load_disk2fold(fold_path)

print(len(training),len(validation))

from tensorflow.keras.callbacks import ReduceLROnPlateau
cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=25, verbose=1, mode='min', min_delta=1e-7, cooldown=1,    
                          min_lr=1e-4)

from tensorflow.keras.callbacks import EarlyStopping
cb_es = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=75, verbose=1, mode='min')


# Run pipeline for cross-validation fold
run_fold(training=training
         , validation=validation
         , fold=fold
         , model=model
         , epochs=1000
         , evaluation_path="evaluation path here"
         , draw_figures=True
         , callbacks=[cb_lr, cb_es]
         , save_models=True
        )