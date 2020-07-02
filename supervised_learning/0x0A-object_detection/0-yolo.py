#!/usr/bin/env python3
""" YOLO """
import tensorflow.keras as K


class Yolo:
    """ Use Yolo v3 algorithm to perform object detection """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ constructor
        @model_path: Where Darknet Keras model is stored
        @classes_path: where the list of class names used for the Darknet
        model, listed in order of index can be found
        @class_t: float representating the box score threshold for the initial
        filtering step
        @nms_t: float representing the IOU threshold for non-max suppression
        @anchors: np.ndarray shape(outputs, anchor_boxes, 2) - all anchor
        boxes
            @outputs: number of outputs (predictions) made by Darknet model
            @anchor_boxes: number of anchor boxes used for each prediction
            @2: => [anchor_box_width, anchor_box_height]
        public instance attributes:
            @model: Darknet Keras model
            @class_names: a list of the class names for the model
            @nms_t: the IOU threshold for non-max suppression
            @anchors: anchor boxes
        """
        # load model
        self.model = K.models.load_model(model_path)
        # get list of class names
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
