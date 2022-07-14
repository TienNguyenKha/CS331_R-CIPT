import os
import random
from pathlib import Path
random.seed(42)

#Chia train và test
def split_mcocr(data_dir = "./ocr_data"):
    
    root = Path(data_dir)
    raw_file_path = root / "text.txt"
    train_file_path = root / "train_annotation.txt"
    val_file_path = root / "val_annotation.txt"

    val_ratio = 0.2


    def writefile(data, filename):
        with open(filename, "w") as f:
            f.writelines(data)


    data = []
    with open(raw_file_path, "r") as f:
        data = f.readlines()
    random.shuffle(data)
    val_len = int(val_ratio * len(data))
    train_data = data[:-val_len]
    val_data = data[-val_len:]
    writefile(train_data, train_file_path)
    writefile(val_data, val_file_path)
# split_mcocr()

# from vietocr.tool.config import Cfg
# from vietocr.model.trainer import Trainer

# config = Cfg.load_config_from_name('vgg_transformer')

# dataset_params = {
#     'name':'mcocr',
#     'data_root':'./ocr_data/',
#     'train_annotation':'train_annotation.txt',
#     'valid_annotation':'val_annotation.txt',
# }

# params = {
#          'print_every':100,
#          'valid_every':500,
#           'iters':20000,
#           'checkpoint':'./checkpoint/transformerocr_checkpoint.pth',    
#           'export':'./weights/transformerocr.pth',
#           'metrics': 3000,
#           'batch_size': 16
#          }

# config['trainer'].update(params)
# config['dataset'].update(dataset_params)
# config['device'] = 'cuda:4'

# trainer = Trainer(config, pretrained=True)
# # trainer.train()
# trainer.config.save('config.yml')



# # bắt đầu huấn luyện 
# trainer.train()
# acc_full_seq, acc_per_char=trainer.precision()
# print(acc_full_seq)
# print(acc_per_char)


# # sử dụng lệnh này để visualize tập train, bao gồm cả augmentation 
# trainer.visualize_dataset()
# # visualize kết quả dự đoán của mô hình
# trainer.visualize_prediction()




from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_seq2seq')

dataset_params = {
    'name':'mcocr',
    'data_root':'./ocr_data/',
    'train_annotation':'train_annotation.txt',
    'valid_annotation':'val_annotation.txt',
}

params = {
         'print_every':100,
         'valid_every':500,
          'iters':20000,
          'checkpoint':'./checkpoint/seq2seqocr_checkpoint.pth',    
          'export':'./weights/seq2seq.pth',
          'metrics': 3000,
          'batch_size': 16
         }

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:4'

trainer = Trainer(config, pretrained=True)
# trainer.train()
trainer.config.save('config_seq2seq.yml')



# bắt đầu huấn luyện 
trainer.train()
acc_full_seq, acc_per_char=trainer.precision()
print(acc_full_seq)
print(acc_per_char)


# sử dụng lệnh này để visualize tập train, bao gồm cả augmentation 
trainer.visualize_dataset()
# visualize kết quả dự đoán của mô hình
trainer.visualize_prediction()

