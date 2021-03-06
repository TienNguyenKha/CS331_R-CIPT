# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse
import torch
from tqdm import tqdm
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
# from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str
from allennlp.data.dataset_readers.dataset_utils.span_utils import InvalidTagSequence

import pathlib
pathlib.WindowsPath=pathlib.PosixPath
from typing import List, Tuple

TypedStringSpan = Tuple[str, Tuple[int, int]]
from transformers import T5ForConditionalGeneration



def bio_tags_to_spans(
        tag_sequence: List[str], text_length: List[int] = None
) -> List[TypedStringSpan]:
    list_idx_to_split = [0]
    init_idx = 0
    for text_len in text_length[0]:
        init_idx += text_len
        list_idx_to_split.append(init_idx)

    spans = []
    line_pos_from_bottom = []
    for index, string_tag in enumerate(tag_sequence):
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]

        if bio_tag == "B":
            if index in list_idx_to_split:
                idx_start = list_idx_to_split.index(index)
                idx_end = list_idx_to_split[idx_start + 1] - 1
                spans.append((conll_tag, (index, idx_end)))
                line_pos_from_bottom.append(idx_start)
    return spans, line_pos_from_bottom
def main_test():
    # args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    # args.add_argument('-ckpt', '--checkpoint', default='PICK-pytorch/model_best.pth', type=str,
    #                   help='path to load checkpoint (default: None)')
    # args.add_argument('--bt', '--boxes_transcripts', default='./boxes_and_transcripts', type=str,
    #                   help='ocr results folder including boxes and transcripts (default: None)')
    # args.add_argument('--impt', '--images_path', default='./images', type=str,
    #                   help='images folder path (default: None)')
    # args.add_argument('-output', '--output_folder', default='./output', type=str,
    #                   help='output folder (default: predict_results)')
    # args.add_argument('-g', '--gpu', default=4, type=int,
    #                   help='GPU id to use. (default: -1, cpu)')
    # args.add_argument('--bs', '--batch_size', default=2, type=int,
    #                   help='batch size (default: 1)')
    # args = args.parse_args()

    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR/mc_ocr/key_info_extraction/PICK/saved/PICK_Default/model_best.pth', type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR/output_2/key_info_extraction/evaluate/boxes_and_transcripts', type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR/output_2/rotation_corrector/evaluate/imgs', type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR/output_2/key_info_extraction/evaluate/txt', type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('-g', '--gpu', default=0, type=int,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=2, type=int,
                      help='batch size (default: 1)')
    args = args.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(560, 784),
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # predict and save to file
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)

            # For easier debug.
            image_names = input_data_item["filenames"]

            output = pick_model(**input_data_item)
            logits = output['logits']  # (B, N*T, out_dim)
            new_mask = output['new_mask']
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            mask = input_data_item['mask']
            # List[(List[int], torch.Tensor)]
            text_length = input_data_item['text_length']
            boxes_coors = input_data_item['boxes_coordinate'].cpu().numpy()[0]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            decoded_texts_list = text_index_to_str(text_segments, mask)

            for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
                # List[ Tuple[str, Tuple[int, int]] ]
                # spans = bio_tags_to_spans(decoded_tags, [])
                spans, line_pos_from_bottom = bio_tags_to_spans(decoded_tags, text_length.cpu().numpy())
                spans = sorted(spans, key=lambda x: x[1][0])

                entities = []  # exists one to many case
                for entity_name, range_tuple in spans:
                    entity = dict(entity_name=entity_name,
                                  text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
                    entities.append(entity)

                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
                with result_file.open(mode='w',encoding='utf-8') as f:
                    for item in entities:
                        f.write('{}\t{}\n'.format(item['entity_name'], item['text']))
                        
# python test.py --checkpoint T:\VNamesereceiptextract\PICK-pytorch\model_best.pth --boxes_transcripts  T:\VNamesereceiptextract\MiAI_Keras_Yolo\boxes_and_transcripts --images_path T:\VNamesereceiptextract\MiAI_Keras_Yolo\images --output_folder T:\VNamesereceiptextract\MiAI_Keras_Yolo\kie_outpu --gpu 0 --batch_size 2
# if __name__ == '__main__':
#     args = argparse.ArgumentParser(description='PyTorch PICK Testing')
#     args.add_argument('-ckpt', '--checkpoint', default='PICK-pytorch/model_best.pth', type=str,
#                       help='path to load checkpoint (default: None)')
#     args.add_argument('--bt', '--boxes_transcripts', default='./boxes_and_transcripts', type=str,
#                       help='ocr results folder including boxes and transcripts (default: None)')
#     args.add_argument('--impt', '--images_path', default='./images', type=str,
#                       help='images folder path (default: None)')
#     args.add_argument('-output', '--output_folder', default='./output', type=str,
#                       help='output folder (default: predict_results)')
#     args.add_argument('-g', '--gpu', default=1, type=int,
#                       help='GPU id to use. (default: -1, cpu)')
#     args.add_argument('--bs', '--batch_size', default=2, type=int,
#                       help='batch size (default: 1)')
#     args = args.parse_args()
#     main(args)

main_test()
