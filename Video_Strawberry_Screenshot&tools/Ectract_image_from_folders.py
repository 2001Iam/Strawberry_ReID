import os
import glob
import shutil
import cv2 as cv

path = r'/home/xplv/fenghao/Market/pytorch/leaved/*'
files = glob.iglob(path)
sorted_files = sorted(files)
for file in sorted_files:
    new_path= file+"/*.jpg"
    new_files = glob.iglob(new_path)
    new_sorted_files = sorted(new_files)
    count = 0
    for new_file in new_sorted_files:
        count += 1
        if (count == 5):
            print(new_file)
            id = new_file[37:41]
            pre_dst_folder = '/home/xplv/fenghao/hhh/pytorch/val'
            save_dir = os.path.join(pre_dst_folder, f'{id}')
            os.makedirs(save_dir, exist_ok=True)
            shutil.move(new_file, save_dir)
            print("success!")
            break