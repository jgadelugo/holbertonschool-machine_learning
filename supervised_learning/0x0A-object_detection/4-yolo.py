#!/usr/bin/env python3
"""
YOLO
"""
import tensorflow.keras as K
import numpy as np
import glob
import cv2


class Yolo:
    """
    Use Yolo v3 algorithm to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        constructor
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

    def sigmoid(self, x):
        """
        sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        processes outputs
        @outputs: list of np.ndarray containing the predictions from Darknet
        model for a single image - shape below
        (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            @grid_height: heigh of ouput grid
            @grid_width: wdith of output grid
            @4: => (t_x, t_y, t_w, t_h)
            @1: => box_confidence
            @classes: class probabilities for all classes
        @image_size: np.ndarray containing the image original
        size [image_height, image_width]
        Returns: tuple (boxes, box_confidences, box_class_probs)
            @boxes: np.ndarray shape (grid_height, grid_width, anchor_boxes,
            4) the processed boundary boxes for each output
                @4: => (x1, y1, x2, y2)
                (x1, y1, x2, y2) represents the boundary box relative to
                original size
            @box_confidence: np.ndarray shape (grid_height, grid_width,
            anchor_boxes, 1) the box confidence for each output
            @box_class_probs: np.ndarray shape (grid_height, grid_width,
            anchor_boxes, classes) box's class probabilities for each output
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height = image_size[0]
        image_width = image_size[1]

        for i, output in enumerate(outputs):
            gh = output.shape[0]
            gw = output.shape[1]

            t_x = output[:, :, :, 0]
            t_y = output[:, :, :, 1]
            t_w = output[:, :, :, 2]
            t_h = output[:, :, :, 3]

            anchor = self.anchors[i]

            pw = anchor[:, 0]
            ph = anchor[:, 1]
            pw = pw.reshape(1, 1, len(pw))
            ph = ph.reshape(1, 1, len(ph))

            cx = np.tile(np.arange(gw), gh).reshape(gw, gw, 1)
            cy = np.tile(np.arange(gw),
                         gh).reshape(gh, gh, 1).T.reshape(gh, gh, 1)

            bx = self.sigmoid(t_x) + cx
            by = self.sigmoid(t_y) + cy

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            # normalize
            bx = bx / gw
            by = by / gh
            bw = bw / int(self.model.input.shape[1])
            bh = bh / int(self.model.input.shape[2])

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.zeros(output[:, :, :, :4].shape)
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2

            boxes.append(box)

            box_confidence = self.sigmoid(output[:, :, :, 4, np.newaxis])
            box_confidences.append(box_confidence)

            box_class = self.sigmoid(output[:, :, :, 5:])
            box_class_probs.append(box_class)
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """ filter boxes method
        @boxes: np.ndarray of shape (grid_height, grid_width, anchor_boxes, 4)
        processed boundary boxes for each output
        @box_confidences: np.ndarray shape(grid_height, grid_width,
        anchor_boxes, 1) processed box confidences for each output
        @box_class_probs: np.ndarray shape(grid_height, grid_width,
        anchor_boxes, classes) processed box class probabilities for each
        Return: (filtered_boxes, box_classes, box_scores)
            @filtered_boxes: np.ndarray (?, 4) class number that each box
            @box_classes: np.ndarray (?,) class number that each box in
            filtered_boxes predicts
            @box_scores: np.ndarray shape(?) the box scores for each box
            in filtered_boxes
        """
        f_boxes = []
        box_classes = []
        b_scores = []

        for i, box_conf in enumerate(box_confidences):
            grid_height, grid_width, anchor_box, _ = box_conf.shape
            for h in range(grid_height):
                for w in range(grid_width):
                    for a_box in range(anchor_box):
                        box = boxes[i][h, w, a_box]
                        classes = box_class_probs[i][h, w, a_box]
                        box_score = box_conf[h, w, a_box, 0] * np.max(classes)
                        if box_score >= .6:
                            f_boxes.append(box)
                            box_classes.append(np.argmax(classes))
                            b_scores.append(box_score)

        return (np.array(f_boxes), np.array(box_classes), np.array(b_scores))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """ non_max_supression method
        filtered_boxes: np.ndarray shpae(?, 4) all of the filtered bounding
        boxes
        box_classes: np.array of shape(?,) class number for the class that
        filtered_boxes predicts
        box_scores: np.ndarray (?) containing the box scores for each box in
        filtered_boxes
        Return: tuple (box_predictions, predicted_box_classes,
        predicted_box_scores)
            @box_predictions: np.ndarray shape(?, 4) predicted bounding boxes
            ordered by class and box score
            @predicted_box_classes: np.ndarray shape(?,) class number for
            box_predictions ordered by class and box score
            @predicted_box_scores: np.ndarray shape(?) box scores for
            box_predictions ordered by class and box score
        """
        box_preds = []
        pred_box_classes = []
        pred_box_scores = []

        u_nums = np.unique(box_classes)

        for i in u_nums:
            idxs = np.where(box_classes == i)

            f_boxes = filtered_boxes[idxs]
            b_scores = box_scores[idxs]
            b_classes = box_classes[idxs]

            idxs = b_scores.argsort()[::-1]

            x1 = f_boxes[:, 0]
            y1 = f_boxes[:, 1]
            x2 = f_boxes[:, 2]
            y2 = f_boxes[:, 3]

            areas = (x2 - x1 + 1) * (y2 - y1 + 1)

            keep = []

            while idxs.size > 0:
                i = idxs[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[idxs[1:]])
                yy1 = np.maximum(y1[i], y1[idxs[1:]])
                xx2 = np.minimum(x2[i], x2[idxs[1:]])
                yy2 = np.minimum(y2[i], y2[idxs[1:]])

                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h

                icu = inter / (areas[i] + areas[idxs[1:]] - inter)

                inds = np.where(icu <= self.nms_t)[0]
                idxs = idxs[inds + 1]

            box_preds.append(f_boxes[keep])
            pred_box_classes.append(b_classes[keep])
            pred_box_scores.append(b_scores[keep])

        box_preds = np.concatenate(box_preds)
        pred_box_classes = np.concatenate(pred_box_classes)
        pred_box_scores = np.concatenate(pred_box_scores)

        return (box_preds, pred_box_classes, pred_box_scores)

    def load_images(self, folder_path):
        """Loads image from directory or file
        @images_path: the path to a directory from which to load images
        @as_array: boolean indicating wether image should be loaded
        as np.ndarray
            *if True: load as np.ndarray of shape (m, h, w, c)
                @m: number of images
                @h, w, c: the height, width, and # of channels of all images
            *if False: load as list of individual np.ndarrays
        * All images should be loaded in RGB format
        * images should be loaded in alphabetical order by filename
        Return: images, filenames
            @images: either a list or np.ndarray of all images
            @filename: list of the filenames assoiated with each
            image in images
        """
        image_paths = glob.glob(folder_path + "/*")

        images = []
        for img in image_paths:
            # converts to np.ndarray
            images.append(cv2.imread(img))

        return images, image_paths
