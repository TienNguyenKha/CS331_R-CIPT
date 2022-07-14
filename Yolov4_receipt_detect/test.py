from PIL import Image
import os
import colorsys

import numpy as np
import tensorflow.keras.backend as K
from keras.models import load_model
from keras.layers import Input

from MiAI_Keras_Yolo.yolo4.model import yolo_eval, yolo4_body
from MiAI_Keras_Yolo.yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import cv2
import torch
from torch import tensor, cat
from keras.utils import multi_gpu_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
class Yolo4(object):
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(608, 608, 3)), num_anchors//3, num_classes)
        self.yolo4_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num>=2:
            self.yolo4_model = multi_gpu_model(self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score)

    def __init__(self, score, iou, anchors_path, classes_path, model_path, gpu_num=4):
        self.score = score
        self.iou = iou
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.load_yolo()

    def close_session(self):
        self.sess.close()

    def detect_image(self, image, model_image_size=(608, 608)):
        start = timer()

        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                # K.learning_phase(): 0 
                # keras_learning_phase:0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='MiAI_Keras_Yolo/font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        labels=[]
        boxes=[]
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{}'.format(predicted_class)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            labels.append(label)
            boxes.append(np.array([left, top, right, bottom]))
            # print(label, (left, top), (right, bottom))
            #bbox_dict[label]=(left, top, right, bottom)
        #return bbox_dict
        boxes = torch.tensor(np.stack(boxes))
        return labels, boxes

        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #     else:
        #         text_origin = np.array([left, top + 1])

        #     # My kingdom for a good redistributable image drawing library.
        #     for i in range(thickness):
        #         draw.rectangle(
        #             [left + i, top + i, right - i, bottom - i],
        #             outline=self.colors[c])
        #     draw.rectangle(
        #         [tuple(text_origin), tuple(text_origin + label_size)],
        #         fill=self.colors[c])
        #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        #     del draw

        # end = timer()
        # print(end - start)
        # return image


# def alignment(detect_result):

#     for i, bbox in enumerate(boxes):
#         bbox = list(map(int, bbox))
#         x_min, y_min, x_max, y_max = bbox
#         cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#         cv2.putText(image, labels[i], (x_min, y_min),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

def non_max_suppression_fast(boxes, labels, overlapThresh=0.15):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    #
    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type

    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")

    return final_boxes, final_labels


def get_center_point(box):
    xmin, ymin, xmax, ymax = box
    # return (xmin + xmax) // 2, (ymin + ymax) // 2
    return xmin + (xmax-xmin)//2, ymin + (ymax-ymin)//2


def perspective_transoform(image, source_points):
    dest_points = np.float32([[0, 0], [350, 0], [350, 800], [0, 800]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (350, 800))

    return dst


if __name__ == '__main__':
    ###################

    # Duong dan den file h5
    model_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/MiAI_Keras_Yolo/yolo4_weight.h5'
    # File anchors cua YOLO
    anchors_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/MiAI_Keras_Yolo/yolo4_anchors.txt'
    # File danh sach cac class
    classes_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/MiAI_Keras_Yolo/yolo.names'

    score = 0.5
    iou = 0.5


    model_image_size = (608, 608)
    yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)

    img = "/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/MiAI_Keras_Yolo/hoadon7.jpg"
    image = Image.open(img)

    # print(result)
    # plt.imshow(result)
    # plt.show()
    labels, boxes=yolo4_model.detect_image(image, model_image_size=model_image_size)
    
    final_boxes, final_labels = non_max_suppression_fast(
        boxes.numpy(), labels, 0.15)
    image=np.array(image)
    final_points = list(map(get_center_point, final_boxes))
    label_boxes = dict(zip(final_labels, final_points))
    yolo4_model.close_session()
    source_points = np.float32([
        label_boxes['top_left'], label_boxes['top_right'], label_boxes['bottom_right'], label_boxes['bottom_left']
    ])

    # Transform
    crop = perspective_transoform(image, source_points)
    cv2.imwrite('filename.jpeg', crop)
  
    
