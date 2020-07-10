#!/usr/bin/env python3
"""Class NST that performs tasks for neural style transfer"""
import numpy as np
import tensorflow as tf


class NST():
    """ Performs task for neural style transfer"""
    # public attributes
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """constructor
        @style_image: image used as style reference, np.ndarray
        @content_image: image used as content ref., np.ndarray
        @alpha: weight for style cost
        @beta: weight for style cost
        """
        # check variables
        err_m1 = "style_image must be a numpy.ndarray with shape (h, w, 3)"
        err_m2 = "content_image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(style_image, np.ndarray):
            raise TypeError(err_m1)
        if len(style_image.shape) != 3 or style_image.shape[2] != 3:
            raise TypeError(err_m1)
        if not isinstance(content_image, np.ndarray):
            raise TypeError(err_m1)
        if len(content_image.shape) != 3 or content_image.shape[2] != 3:
            raise TypeError(err_m1)
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    def scale_image(self, image):
        """rescales image such its pixels values are between 0-1 and
        largest side is 512 pixels
        @image: image to scape, np.ndarray shape(h, w, 3)
        Return: scaled image
        """
        err_m1 = "image must be a numpy.ndarray with shape (h, w, 3)"
        if not isinstance(image, np.ndarray):
            raise TypeError(err_m1)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise TypeError(err_m1)
        shape0 = image.shape[0]
        shape1 = image.shape[1]

        if shape0 > shape1:
            new_w = int(image.shape[1] * 512 / image.shape[0])
            new_h = 512
        elif shape0 < shape1:
            new_h = int(image.shape[0] * 512 / image.shape[1])
            new_w = 512
        else:
            new_h = 512
            new_w = 512
        tf.image.ResizeMethod.BICUBIC
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_bicubic(image,
                                        (new_h, new_w),
                                        align_corners=False)
        image = image / 255
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)

        return image

    def load_model(self):
        """ load VGG19 mdoel"""
        # Load our model. We load pretrained VGG, trained on imagenet data (weights=’imagenet’)
        vgg = tf.keras.applications.vgg19.VGG19(include_top=False,
                                                weights='imagenet',
                                                pooling='avg')
        vgg.save("base_model")
        co = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model("base_model",
                                         custom_objects=co)
        for layer in vgg.layers:
            layer.trainable = False
        # Get output layers corresponding to style and content layers 
        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [vgg.get_layer(self.content_layer).output]
        model_outputs = style_outputs + content_outputs
        # Build model 
        self.model = tf.keras.models.Model(vgg.input, model_outputs)
