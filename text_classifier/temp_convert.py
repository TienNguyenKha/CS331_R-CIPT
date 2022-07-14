anno_path='/mlcv/WorkingSpace/Personals/tiennk/MC_OCR_compose/text_classifier/cls_out_txt/filename.txt'

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