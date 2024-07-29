import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

x_index = []
y_index = []
input_file = 'L1_2.txt'
fig, ax = plt.subplots()
count = 0
with open(input_file, 'r') as file:
    for line in file:
        count += 1
        if count % 10 == 0:
            parts = line.strip().split(',')
            frame_num = int(parts[0])
            bbox_class = parts[1]
            bbox_id = int(parts[2])
            x = int(parts[3])
            y = int(parts[4])
            w = int(parts[5])
            h = int(parts[6])

            left = 1500 - w / 2
            bottom = 800 - h / 2
            rectangle = mpatches.Rectangle((left, bottom), w, h,
                                           fill=False, color='blue', alpha=0.05)
            ax.add_patch(rectangle)

ax.set_xlabel('x', fontsize=14)
ax.set_ylabel('y', fontsize=14)
ax.axis('equal')
# 显示图形
plt.show()
fig.savefig('show.png')

"""x0 = 1500
        y0 = 800
        x1 = x0 - w / 2
        y1 = y0 + h / 2
        x2 = x0 + w / 2
        y2 = y0 - h / 2
        plt.plot([x1, x2], [y1, y1], 'b', lw=2, alpha=0.1)  # 上边
        plt.plot([x1, x1], [y1, y2], 'b', lw=2, alpha=0.1)  # 左边
        plt.plot([x2, x2], [y1, y2], 'b', lw=2, alpha=0.1)  # 右边
        plt.plot([x1, x2], [y2, y2], 'b', lw=2, alpha=0.1)  #
"""

'''ax.set_xlim([1400, 1600])  # 横轴范围
ax.set_ylim([700, 900])  # 纵轴范围
# 设置刻度，使它们的比例一致
ax.set_xticks(np.arange(1400, 1601, 25))
ax.set_yticks(np.arange(700, 901, 25))'''
