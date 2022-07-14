import os
import cv2
from pathlib import Path
import numpy as np
from datetime import datetime

os.environ["PYTHONIOENCODING"] = "utf-8"
pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

def get_list_boxes_from_icdar(anno_path):
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


# import the necessary packages
from scipy.spatial import distance as dist
import numpy as np
import cv2

def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

images_dir='./rotation_corrector/mc_ocr_train_filtered/imgs'
txt_dir='./text_classifier/mc_ocr_train_filtered/txt'
output_ocr="./ocr_data"

images_list=os.listdir(images_dir)
count=0
with open(Path(output_ocr)/"text.txt","w", encoding='utf8') as f:
    for i in images_list:
        abs_img_dir=os.path.join(images_dir,i)
        abs_txt_dir=os.path.join(txt_dir,i.replace('.jpg','.txt'))

        img=cv2.imread(abs_img_dir)

        boxes_list,texts_list=get_list_boxes_from_icdar(abs_txt_dir)
        for box,text in zip(boxes_list,texts_list):
            if len(text.rstrip())<=1:
                continue
            temp=np.array(box, dtype = "float32").reshape(4,2)
            cv2.imwrite('./ocr_data/imgs/{}.jpg'.format(count), four_point_transform(img,np.array(temp, dtype = "float32")))
            txt_line='imgs/{}.jpg'.format(count)+'\t'+text+'\n'
            f.write(txt_line)
            count+=1

print(count)




# abs_img_dir='./rotation_corrector/mc_ocr_train_filtered/imgs/mcocr_public_145014cgbau.jpg'
# abs_txt_dir='./text_classifier/mc_ocr_train_filtered/txt/mcocr_public_145014cgbau.txt'

# img=cv2.imread(abs_img_dir)
# boxes_list,text=get_list_boxes_from_icdar(abs_txt_dir)
# # print(boxes_list[2])
# # temp=[268, 919, 427, 921, 427, 943, 267, 941]
# temp=np.array(boxes_list[2], dtype = "float32").reshape(4,2)
# print(text[2])  
# cv2.imwrite('test.jpg', four_point_transform(img,np.array(temp, dtype = "float32")))
  

