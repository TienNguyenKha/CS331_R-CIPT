from Yolov4_receipt_detect.test import Yolo4,non_max_suppression_fast
from PIL import Image
import tensorflow as tf
import cv2
import numpy as np

model_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/Yolov4_receipt_detect/yolo4_weight.h5'
anchors_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/Yolov4_receipt_detect/yolo4_anchors.txt'
classes_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/Yolov4_receipt_detect/yolo.names'

score = 0.5
iou = 0.5

# sess=tf.Session()
# graph=tf.get_default_graph()

model_image_size = (608, 608)
# with sess.as_default():
#     with graph.as_default():
yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)

#Input image
filename = 'filename.jpeg'
# cv2.imwrite(filename,img)
image = Image.open(filename)

labels, boxes=yolo4_model.detect_image(image, model_image_size=model_image_size)
final_boxes, final_labels = non_max_suppression_fast(
    boxes.numpy(), labels, 0.15)
box=final_boxes[0]
print(box)
image=np.array(image)
image_croped=image[box[1]:box[3],box[0]:box[2]]
cv2.imwrite(filename,image_croped)