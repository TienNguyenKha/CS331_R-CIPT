import cv2, os, time
from datetime import datetime
from vietocr.vietocr_class import Classifier_Vietocr
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
cls_visualize = True
cls_out_txt_dir='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_classifier/cls_out_txt'
cls_out_viz_dir='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_classifier/cls_out_viz'
gpu='0'
# img_dir = rot_out_img_dir
img_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/MiAI_Keras_Yolo/filename.jpeg'
# anno_dir = rot_out_txt_dir
anno_path = '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_detector/det_out_txt/filename.txt'

write_file = True

class poly():
    def __init__(self, segment_pts, type=1, value=''):
        if isinstance(segment_pts, str):
            segment_pts = [int(f) for f in segment_pts.split(',')]
        elif isinstance(segment_pts, list):
            segment_pts = [round(f) for f in segment_pts]
        self.type = type
        self.value = value
        num_pts = int(len(segment_pts) / 2)
        # print('num_pts', num_pts)
        first_pts = [segment_pts[0], segment_pts[1]]
        self.list_pts = [first_pts]
        for i in range(1, num_pts):
            self.list_pts.append([segment_pts[2 * i], segment_pts[2 * i + 1]])

    def reduce_pts(self, dist_thres=7):  # reduce nearly duplicate points
        last_pts = self.list_pts[0]
        filter_pts = []
        for i in range(1, len(self.list_pts)):
            curr_pts = self.list_pts[i]
            dist = euclidean_distance(last_pts, curr_pts)
            # print('distance between', i - 1, i, ':', dist)
            if dist > dist_thres:
                filter_pts.append(last_pts)
                print('Keep point', i - 1)
            last_pts = curr_pts

        # print('distance between', len(self.list_pts) - 1, 0, ':', euclidean_distance(last_pts, self.list_pts[0]))
        if euclidean_distance(last_pts, self.list_pts[0]) > dist_thres:
            filter_pts.append(last_pts)
            # print('Keep last point')

        self.list_pts = filter_pts

    def check_max_wh_ratio(self):
        max_ratio = 0
        if len(self.list_pts) == 4:
            first_edge = euclidean_distance(self.list_pts[0], self.list_pts[1])
            second_edge = euclidean_distance(self.list_pts[1], self.list_pts[2])
            if first_edge / second_edge > 1:
                long_edge = (self.list_pts[0][0] - self.list_pts[1][0], self.list_pts[0][1] - self.list_pts[1][1])
            else:
                long_edge = (self.list_pts[1][0] - self.list_pts[2][0], self.list_pts[1][1] - self.list_pts[2][1])
            max_ratio = max(first_edge / second_edge, second_edge / first_edge)
        else:
            print('check_max_wh_ratio. Polygon is not qualitareal')
        return max_ratio, long_edge

    def check_horizontal_box(self):
        if len(self.list_pts) == 4:
            max_ratio, long_edge = self.check_max_wh_ratio()
            if long_edge[0] == 0:
                angle_with_horizontal_line = 90
            else:
                angle_with_horizontal_line = math.atan2(long_edge[1], long_edge[0]) * 57.296
        else:
            print('check_horizontal_box. Polygon is not qualitareal')
        print('Angle', angle_with_horizontal_line)
        if math.fabs(angle_with_horizontal_line) > 45 and math.fabs(angle_with_horizontal_line) < 135:
            return False
        else:
            return True

    def get_horizontal_angle(self):
        assert len(self.list_pts) == 4
        max_ratio, long_edge = self.check_max_wh_ratio()
        if long_edge[0] == 0:
            if long_edge[1] < 0:
                angle_with_horizontal_line = -90
            else:
                angle_with_horizontal_line = 90
        else:
            angle_with_horizontal_line = math.atan2(long_edge[1], long_edge[0]) * 57.296
        return angle_with_horizontal_line

    def to_icdar_line(self, map_type=None):
        line_str = ''
        if len(self.list_pts) == 4:
            for pts in self.list_pts:
                line_str += '{},{},'.format(pts[0], pts[1])
            if map_type is not None:
                line_str += self.value + ',' + str(map_type[self.type])
            else:
                line_str += self.value + ',' + str(self.type)

        else:
            print('to_icdar_line. Polygon is not qualitareal')
        return line_str
def viz_poly(img, list_poly, save_viz_path=None, ignor_type=[1]):
    '''
    visualize polygon
    :param img: numpy image read by opencv
    :param list_poly: list of "poly" object that describe in common.py
    :param save_viz_path:
    :return:
    '''
    fig, ax = plt.subplots(1)
    fig.set_size_inches(20, 20)
    plt.imshow(img)

    for polygon in list_poly:
        ax.add_patch(
            patches.Polygon(polygon.list_pts, linewidth=2, edgecolor=color_map[polygon.type], facecolor='none'))
        draw_value = polygon.value
        if polygon.type in ignor_type:
            draw_value = ''
        plt.text(polygon.list_pts[0][0], polygon.list_pts[0][1], draw_value, fontsize=20,
                 fontdict={"color": txt_color_map[polygon.type]})
    # plt.show()

    if save_viz_path is not None:
        print('Save visualized result to', save_viz_path)
        fig.savefig(save_viz_path, bbox_inches='tight')

def viz_icdar(img_path, anno_path, save_viz_path=None, extract_kie_type=False, ignor_type=[1]):
    if not isinstance(img_path, str):
        image = img_path
    else:
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    list_poly = []
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()

    for anno in anno_txt:
        anno = anno.rstrip('\n')

        idx = -1
        for i in range(0, 8):
            idx = anno.find(',', idx + 1)

        coordinates = anno[:idx]
        val = anno[idx + 1:]
        type = 1
        if extract_kie_type:
            last_comma_idx = val.rfind(',')
            type_str = val[last_comma_idx + 1:]
            val = val[:last_comma_idx]
            if type_str in inv_type_map.keys():
                type = inv_type_map[type_str]

        coors = [int(f) for f in coordinates.split(',')]
        pol = poly(coors, type=type, value=val)
        list_poly.append(pol)
    viz_poly(img=image,
             list_poly=list_poly,
             save_viz_path=save_viz_path,
             ignor_type=ignor_type)

def rotate_and_crop(img, points, debug=False, rotate=True, extend=True,
                    extend_x_ratio=1, extend_y_ratio=0.01,
                    min_extend_y=1, min_extend_x=2):
    rect = cv2.minAreaRect(points)

    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    if debug:
        print("shape of cnt: {}".format(points.shape))
        print("rect: {}".format(rect))
        print("bounding box: {}".format(box))
    # cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    # get width and height of the detected rectangle
    height = int(rect[1][0])
    width = int(rect[1][1])

    if extend:
        if width > height:
            w, h = width, height
        else:
            h, w = width, height
        ex = min_extend_x if (extend_x_ratio * w) < min_extend_x else (extend_x_ratio * w)
        ey = min_extend_y if (extend_y_ratio * h) < min_extend_y else (extend_y_ratio * h)
        ex = int(round(ex))
        ey = int(round(ey))
        if width < height:
            ex, ey = ey, ex
    else:
        ex, ey = 0, 0
    src_pts = box.astype("float32")
    # width = width + 10
    # height = height + 10
    dst_pts = np.array([
        [width - 1 + ex, height - 1 + ey],
        [ex, height - 1 + ey],
        [ex, ey],
        [width - 1 + ex, ey]
    ], dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    # print(M)
    warped = cv2.warpPerspective(img, M, (width + 2 * ex, height + 2 * ey))
    h, w, c = warped.shape
    rotate_warped = warped
    if w < h and rotate:
        rotate_warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    if debug:
        print('ex, ey', ex, ey)
        cv2.imshow('before rotated', warped)
        cv2.imshow('rotated', rotate_warped)
        cv2.waitKey(0)
    return rotate_warped

def init_models(gpu='0'):
    if gpu != None:
        print('Use GPU', gpu)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    else:
        print('Use CPU')
    classifier = Classifier_Vietocr(gpu = gpu)
    return classifier


def get_boxes_data(img_data, boxes, extend_box=True,
                   extend_y_ratio=0.05,
                   min_extend_y=1,
                   extend_x_ratio=0.05,
                   min_extend_x=2):
    boxes_data = []
    for box_loc in boxes:
        box_loc = np.array(box_loc).astype(np.int32).reshape(-1, 1, 2)
        box_data = rotate_and_crop(img_data, box_loc, debug=False, extend=extend_box,
                                   extend_x_ratio=extend_x_ratio, extend_y_ratio=extend_y_ratio,
                                   min_extend_y=min_extend_y, min_extend_x=min_extend_x)
        boxes_data.append(box_data)
    return boxes_data


def get_list_boxes_from_icdar(anno_path):
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno_txt = f.readlines()
    list_boxes = []
    for anno in anno_txt:
        anno = anno.rstrip('\n')

        idx = -1
        for i in range(0, 8):
            idx = anno.find(',', idx + 1)

        coordinates = anno[:idx]

        coors = [int(f) for f in coordinates.split(',')]
        list_boxes.append(coors)
    return list_boxes


def main():
    begin_init = time.time()
    global anno_path
    classifier = init_models(gpu=gpu)
    end_init = time.time()
    print('Init models time:', end_init - begin_init, 'seconds')
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

        # 2 extend y by 10%
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


def write_output(list_boxes, values, probs, result_file_path, prob_thres=0.7):
    result = ''
    for idx, box in enumerate(list_boxes):
        s = [str(i) for i in box]
        if probs[idx] > prob_thres:
            line = ','.join(s) + ',' + values[idx]
        else:
            line = ','.join(s) + ','
        result += line + '\n'
    result = result.rstrip('\n')

    with open(result_file_path, 'w', encoding='utf8') as res:
        res.write(result)


if __name__ == '__main__':
    # os.environ["DISPLAY"] = ":11.0"
    main()