from text_detector.PaddleOCR.tools.infer.self_predict_det import TextDetector
import text_detector.PaddleOCR.tools.infer.utility as utility
from text_detector.PaddleOCR.ppocr.utils.logging import get_logger
from text_detector.PaddleOCR.ppocr.utils.utility import get_image_file_list, check_and_read_gif
from text_detector.PaddleOCR.ppocr.data import create_operators, transform
from text_detector.PaddleOCR.ppocr.postprocess import build_post_process
import os
import cv2

args = utility.parse_args()

# link model paddle
det_model_dir ='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_detector/PaddleOCR/inference/ch_ppocr_server_v2.0_det_infer'
det_visualize = True
det_db_thresh = 0.3
det_db_box_thresh = 0.3

det_out_viz_dir= '/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_detector/det_out_viz'
det_out_txt_dir='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_detector/det_out_txt'



args.det_model_dir = det_model_dir
args.det_db_thresh = det_db_thresh
args.det_db_box_thresh = det_db_box_thresh
args.use_gpu = False

# print(args)

text_detector = TextDetector(args)

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