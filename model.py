# Imports

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input, UpSampling2D, 
                                    add, multiply, MaxPooling2D, Dense




# Define network architecture

class Architecture():
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, activation='softmax'):
        # Parse activation
        self.activation = activation

    #---------------------------------------------#
    #               Create 2D Model               #
    #---------------------------------------------#
    def create_model_2D(self, input_shape, n_labels=2):

        # Input layer
        inputs = Input(input_shape) 

        encoder = ResNet152V2(include_top=False, weights="imagenet", input_tensor=inputs)
        encoder.trainable = False

        t1 = encoder.get_layer("input_1").output    
        t2 = encoder.get_layer("conv1_conv").output  
        t3 = encoder.get_layer("conv2_block3_preact_relu").output 
        t4 = encoder.get_layer("conv3_block8_preact_relu").output 
        t5 = encoder.get_layer("conv4_block36_preact_relu").output 

        t1 = conv_block(t1,32)
        t2 = conv_block(t2,64)
        t3 = conv_block(t3,128)
        t4 = conv_block(t4,256)
        t5 = conv_block(t5,512)

        g = GatingSignal(t5, t4)

        d1 = decoder_block(t5, t4, 256, g)                               
        d2 = decoder_block(d1, t3, 128, g)   
        d3 = decoder_block(d2, t2, 64, g)           
        d4 = decoder_block(d3, t1, 32, g)          

        """ Output """
        outputs = Conv2D(2, 1, padding="same", activation=self.activation)(d4)

        model = Model(inputs, outputs, name="AttnUNet")
        return model  

    def create_model_3D(self, input_shape, n_labels=2):
        pass

# Define layers

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = tfa.layers.InstanceNormalization()(x)
    x1 = Activation('relu')(x)

    x2 = Conv2D(num_filters, 3, padding="same")(x1)
    x2 = tfa.layers.InstanceNormalization()(x2)
    x2 = Activation('relu')(x2)

    out = Concatenate()([x1,x2])

    return out

def decoder_block(inputs, skip, num_filters, gating):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    attn = AttentionGate(gating, skip)
    x = Concatenate()([x, attn])
    x = conv_block(x, num_filters)
    return x


def AttentionGate(input, skip_connections):

    resized_input = Conv2D(skip_connections.shape[-1], 1, strides=(1, 1), padding='same', use_bias=True)(input)
    # Resize skip connections to match image shape for input
    resized_skip = Conv2D(skip_connections.shape[-1], 1, strides=(2, 2), padding='same', use_bias=False)(skip_connections)

    stride_x = resized_skip.shape[1] // input.shape[1]
    stride_y = resized_skip.shape[2] // input.shape[2]
      
    resized_input = UpSampling2D((stride_x,stride_y))(resized_input)

    # element wise addition
    sum  = add([resized_input, resized_skip])

    # perform non-linear activation
    act = Activation('relu')(sum)

    weights = Conv2D(1, kernel_size=1, padding="same", use_bias = True)(act)

    weights = Activation('sigmoid')(weights)

    # upsample to match skip connection image shape 
    stride_x_weights = skip_connections.shape[1] // weights.shape[1]
    stride_y_weights = skip_connections.shape[2] // weights.shape[2]
    weights = UpSampling2D((stride_x_weights, stride_y_weights))(weights) 

    # multiply skip connections by weights
    output = multiply([weights, skip_connections])

    return output

def GatingSignal(input, skip_connections):
    signal = Conv2D(skip_connections.shape[-1], (1, 1), strides=(1, 1), padding="same")(input)
    signal = tfa.layers.InstanceNormalization()(signal)
    signal = Activation('relu')(signal)
    return signal