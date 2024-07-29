# Author:fenghao
# CreatTime: 2024/7/2
# FileName: Video_strawberry_Screenshot
# Description: simple introduction of the code
import cv2 as cv
import glob
import os
import matplotlib.pyplot as plt



x_index=[]
y_index=[]
path=r'*_2.txt'
files = glob.iglob(path)
for input_file in files:
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame_num = int(parts[0])
            bbox_class = parts[1]
            bbox_id = int(parts[2])
            x = int(parts[3])
            y = int(parts[4])
            w = int(parts[5])
            h = int(parts[6])

            center_x=x+w/2
            center_y=y+h/2
            x_index.append(center_x)
            y_index.append(center_y)
            print(f'({center_x},{center_y})')


fig, ax = plt.subplots()
ax.scatter(x_index,y_index,color='blue',alpha=0.3,s=1)
# 设置坐标轴标签
ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.axis('equal')
# 显示图形
plt.show()
fig.savefig('center_coordinate.png')