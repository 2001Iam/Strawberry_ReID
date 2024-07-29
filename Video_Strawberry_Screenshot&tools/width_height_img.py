import os
import glob
import shutil
import cv2 as cv
import matplotlib.pyplot as plt

path = r'/home/xplv/fenghao/Video_Strawberry_Screenshot/save_pic/*'
files = glob.iglob(path)
sorted_files = sorted(files)
widths=[]
heights=[]
for file in sorted_files:
    new_path = file + "/*.jpg"
    new_files = glob.iglob(new_path)
    new_sorted_files = sorted(new_files)
    for new_file in new_sorted_files:
        img = cv.imread(new_file)
        w, h = img.shape[:2]
        widths.append(w)
        heights.append(h)
fig, ax = plt.subplots()
ax.scatter(widths, heights,color='blue',alpha=0.5,s=5)
# 设置坐标轴标签
ax.set_xlabel('width', fontsize=14)
ax.set_ylabel('height', fontsize=14)
ax.axis('equal')
# 显示图形
plt.show()