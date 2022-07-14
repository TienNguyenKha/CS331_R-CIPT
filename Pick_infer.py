import numpy as np
import cv2
import base64
import os
from PIL import Image
import csv

def get_list_boxes_from_icdar_2(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()
    list_boxes = []
    text=[]
    for anno in anno_txt:
        anno = anno.rstrip('\n')

        idx = -1
        for i in range(0, 8):
            idx = anno.find(',', idx + 1)

        coordinates = anno[:idx]

        coors = [int(f) for f in coordinates.split(',')]
        list_boxes.append(coors)
        text.append(anno[idx+1:])
    return list_boxes,text
#prepare data for PICK model:
with open('boxes_and_transcripts/filename.tsv', 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    # tsv_writer.writerow(['name', 'field'])
    # img = Image.open(img_path)
    idx=1
    lst_boxes,texts_list=get_list_boxes_from_icdar_2('text_classifier/cls_out_txt/filename.txt')
    for box,text in zip(lst_boxes,texts_list):
        x1,y1,x2,y2,x3,y3,x4,y4=box
        writer.writerow([idx, x1,y1,x2,y2,x3,y3,x4,y4,text])
        idx+=1

#Pick infer:
from PICK-pytorch.test import main_test
main_test()
