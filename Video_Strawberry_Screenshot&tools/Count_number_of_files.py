# Author:fenghao
# CreatTime: 2024/7/2
# FileName: Video_strawberry_Screenshot
# Description: simple introduction of the code
import os
import glob
import shutil

def count_files_in_folder(folder_path):
    try:
        # 使用os.listdir获取指定路径下的所有文件和子目录名称
        files = os.listdir(folder_path)

        # 过滤掉非文件（如子目录）并只保留文件名
        files = [file for file in files if os.path.isfile(os.path.join(folder_path, file))]

        # 返回文件数量
        return len(files)

    except FileNotFoundError:
        print(f"指定的文件夹 {folder_path} 不存在.")
        return 0

    except PermissionError:
        print("没有足够的权限访问该文件夹.")
        return 0

# 使用方法
path = r'/home/xplv/fenghao/hhh/pytorch/save_pic/*'
files = glob.iglob(path)
sorted_files = sorted(files)
for folder_to_check in sorted_files:
    num_of_files = count_files_in_folder(folder_to_check)
    if num_of_files<11:
        print(f"{folder_to_check}文件夹中有 {num_of_files} 个文件.")
        dst_folder = '/home/xplv/fenghao/hhh/pytorch/num_lower_10'
        shutil.move(folder_to_check, dst_folder)
        print("success!")