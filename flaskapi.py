from flask import Flask
from flask_cors import CORS, cross_origin
from flask import request

import numpy as np
import cv2
import base64
import os
############YOLOV4##################
from MiAI_Keras_Yolo.test import Yolo4,non_max_suppression_fast
from PIL import Image
import tensorflow as tf

model_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/MiAI_Keras_Yolo/yolo4_weight.h5'
anchors_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/MiAI_Keras_Yolo/yolo4_anchors.txt'
classes_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/MiAI_Keras_Yolo/yolo.names'

score = 0.5
iou = 0.5

# sess=tf.Session()
# graph=tf.get_default_graph()

model_image_size = (608, 608)
# with sess.as_default():
#     with graph.as_default():
yolo4_model = Yolo4(score, iou, anchors_path, classes_path, model_path)

######################PADDLE#######################################
from text_detector.PaddleOCR.tools.infer.self_predict_det import TextDetector
import text_detector.PaddleOCR.tools.infer.utility as utility
from text_detector.PaddleOCR.ppocr.utils.logging import get_logger
from text_detector.PaddleOCR.ppocr.utils.utility import get_image_file_list, check_and_read_gif
from text_detector.PaddleOCR.ppocr.data import create_operators, transform
from text_detector.PaddleOCR.ppocr.postprocess import build_post_process

args = utility.parse_args()

# det_model_dir = full_path('text_detector/PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer')
det_model_dir ='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_detector/PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer'
det_visualize = True
det_db_thresh = 0.3
det_db_box_thresh = 0.3
# det_out_viz_dir = output_path('text_detector/{}/viz_imgs'.format(dataset))
det_out_viz_dir= '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_detector/det_out_viz'
det_out_txt_dir='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_detector/det_out_txt'
# det_out_txt_dir = output_path('text_detector/{}/txt'.format(dataset))


args.det_model_dir = det_model_dir
args.det_db_thresh = det_db_thresh
args.det_db_box_thresh = det_db_box_thresh
args.use_gpu = False

# print(args)

text_detector = TextDetector(args)

###########VIETOCR#################
import text_classifier.self_pred_ocr
from text_classifier.self_pred_ocr import poly,viz_poly,viz_icdar,rotate_and_crop,init_models,get_boxes_data,get_list_boxes_from_icdar,write_output
import cv2, os, time
from datetime import datetime
import numpy as np

#for visualize
import matplotlib
matplotlib.rc('font', family='TakaoPGothic')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

type_map = {1: 'OTHER', 15: 'SELLER', 16: 'ADDRESS', 17: 'TIMESTAMP', 18: 'TOTAL_COST'}
color_map = {1: 'r', 15: 'green', 16: 'blue', 17: 'm', 18: 'cyan'}
txt_color_map = {1: 'b', 15: 'green', 16: 'blue', 17: 'm', 18: 'cyan'}
inv_type_map = {v: k for k, v in type_map.items()}

os.environ["PYTHONIOENCODING"] = "utf-8"
pred_time = datetime.today().strftime('%Y-%m-%d_%H-%M')

# config model and path
cls_ocr_thres = 0.65
cls_visualize = False
cls_out_txt_dir='text_classifier/cls_out_txt'
cls_out_viz_dir='text_classifier/cls_out_viz'
gpu='2'

write_file = True

begin_init = time.time()
global anno_path
classifier = init_models(gpu=gpu)
end_init = time.time()

app = Flask(__name__)
# Apply Flask CORS
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

import csv


def chuyen_base64_sang_anh(anh_base64):
    try:
        anh_base64 = np.fromstring(base64.b64decode(anh_base64), dtype=np.uint8)
        anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return anh_base64


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


############API###################
print('Init models time:', end_init - begin_init, 'seconds')
@app.route('/detect2ocr', methods=['POST'] )
@cross_origin(origin='*')
def Yolo2VietOCR():
    # Đọc ảnh từ client gửi lên
    
    # imgbase64 = request.form.get('imgbase64')
    # # Chuyển base 64 về OpenCV Format
    # img = chuyen_base64_sang_anh(imgbase64)
    # #Detect:
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
    # Save for PICK
    cv2.imwrite('images/'+filename,image_croped)
    ####text_detect#####
    count = 0
    total_time = 0
    image_file='filename.jpeg'
    img, flag = check_and_read_gif(image_file)
    if not flag:
        img = cv2.imread(image_file)
    
    dt_boxes, elapse = text_detector(img)
    if count > 0:
        total_time += elapse
    count += 1
    

    
    img_name_pure = os.path.split(image_file)[-1]
    output_txt_path = os.path.join(det_out_txt_dir, img_name_pure.replace('.jpeg', '.txt'))
    src_im = utility.draw_text_det_res(dt_boxes, image_file, save_path=output_txt_path)
    if det_visualize:
        img_path = os.path.join(det_out_viz_dir, "{}".format(img_name_pure))
        cv2.imwrite(img_path, src_im)
    
    ###########VietOCR####################
    img_path = 'filename.jpeg'
    anno_path = 'text_detector/det_out_txt/filename.txt'

    begin = time.time()
    list_img_path = []
    list_img_path.append(img_path)
    
    for idx, img_name in enumerate(list_img_path):
        if idx < 0:
            continue
        print('\n', idx, 'Inference', img_name)

        test_img = cv2.imread(img_name)
        begin_detector = time.time()

        boxes_list = get_list_boxes_from_icdar(anno_path)

        end_detector = time.time()
        print('get boxes from icdar time:', end_detector - begin_detector, 'seconds')

        # multiscale ocr

        list_values = []
        list_probs = []
        total_boxes = len(boxes_list)

        # 1 Extend x, no extend y
        boxes_data = get_boxes_data(test_img, boxes_list, extend_box=True, min_extend_y=0, extend_y_ratio=0)
        values, probs = classifier.inference(boxes_data, debug=False)
        list_values.append(values)
        list_probs.append(probs)

        # # 2 extend y by 10%
        # boxes_data = get_boxes_data(test_img, boxes_list, extend_box=True, min_extend_y=2, extend_y_ratio=0.1)
        # values, probs = classifier.inference(boxes_data, debug=False)
        # list_values.append(values)
        # list_probs.append(probs)

        # # 3 extend y by 20%
        # boxes_data = get_boxes_data(test_img, boxes_list, extend_box=True, min_extend_y=4, extend_y_ratio=0.2)
        # values, probs = classifier.inference(boxes_data, debug=False)
        # list_values.append(values)
        # list_probs.append(probs)

        # combine final values and probs
        final_values = []
        final_probs = []
        for idx in range(total_boxes):
            max_prob = list_probs[0][idx]
            max_value = list_values[0][idx]
            for n in range(1, len(list_values)):
                if list_probs[n][idx] > max_prob:
                    max_prob = list_probs[n][idx]
                    max_value = list_values[n][idx]

            final_values.append(max_value)
            final_probs.append(max_prob)

        end_classifier = time.time()
        print('Multiscale OCR time:', end_classifier - end_detector, 'seconds')
        print('Total predict time:', end_classifier - begin_detector, 'seconds')
        output_txt_path = os.path.join(cls_out_txt_dir, os.path.basename(img_name).split('.')[0] + '.txt')
        output_viz_path = os.path.join(cls_out_viz_dir, os.path.basename(img_name))
        if write_file:
            write_output(boxes_list, final_values, final_probs, output_txt_path, prob_thres=cls_ocr_thres)

        if cls_visualize:
            viz_icdar(img_name, output_txt_path, output_viz_path, ignor_type=[])
            end_visualize = time.time()
            print('Visualize time:', end_visualize - end_classifier, 'seconds')

    end = time.time()
    speed = (end - begin) / len(list_img_path)
    print('\nTotal processing time:', end - begin, 'seconds. Speed:', round(speed, 4), 'second/image')


    #prepare data for PICK model:
    with open('boxes_and_transcripts/filename.tsv', 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        # tsv_writer.writerow(['name', 'field'])
        img = Image.open(img_path)
        idx=1
        lst_boxes,texts_list=get_list_boxes_from_icdar_2('text_classifier/cls_out_txt/filename.txt')
        for box,text in zip(lst_boxes,texts_list):
            x1,y1,x2,y2,x3,y3,x4,y4=box
            writer.writerow([idx, x1,y1,x2,y2,x3,y3,x4,y4,text])
            idx+=1
    

    return 'done'


    


# Start Backend
if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000')