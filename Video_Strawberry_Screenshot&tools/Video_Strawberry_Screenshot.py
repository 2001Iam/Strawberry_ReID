# Author:fenghao
# CreatTime: 2024/7/2
# FileName: Video_strawberry_Screenshot
# Description: simple introduction of the code
import cv2 as cv
import glob
import os


path = r'*.mp4'

files = glob.iglob(path)
count = -1
for file in files:
    mp = {}
    pre = file[0:2]
    input_file = pre + "_2.txt"
    cap = cv.VideoCapture(file)
    # timef = 75
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            # 处理每一帧，例如保存到文件或显示
            frameNum = cap.get(cv.CAP_PROP_POS_FRAMES)
            frameNum -= 1
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
                    if bbox_id in mp:
                        res = mp[bbox_id]
                    else:
                        count +=1
                        mp[bbox_id] = count
                        res = count
                    if (frameNum == frame_num):  # 如果是就截取
                        # x1 = x - w // 2
                        # x2 = x + w // 2
                        # y1 = y - h // 2
                        # y2 = y + h // 2
                        imgCrop = frame[y:y + h, x:x + w].copy()
                        cv.imshow("demo", imgCrop)
                        save_dir = os.path.join('save_pic', f'{res:04d}')
                        os.makedirs(save_dir, exist_ok=True)
                        filename = f'{res:04d}_L2_2_{bbox_class}{bbox_id}_{frame_num}frame.jpg'
                        cv.imwrite(os.path.join(save_dir, filename), imgCrop)
            # 按 'q' 键停止
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv.destroyAllWindows()
