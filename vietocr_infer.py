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
    boxes_data = get_boxes_data(test_img, boxes_list, extend_box=True, min_extend_y=2, extend_y_ratio=0.1)
    values, probs = classifier.inference(boxes_data, debug=False)
    list_values.append(values)
    list_probs.append(probs)

    # 3 extend y by 20%
    boxes_data = get_boxes_data(test_img, boxes_list, extend_box=True, min_extend_y=4, extend_y_ratio=0.2)
    values, probs = classifier.inference(boxes_data, debug=False)
    list_values.append(values)
    list_probs.append(probs)

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