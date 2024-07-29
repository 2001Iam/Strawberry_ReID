# Strawberry_ReID

## Trained Model

|                Methods                 | Rank@1 |  mAP  | Reference                                                    |
| :------------------------------------: | :----: | :---: | ------------------------------------------------------------ |
|              [ResNet-50]               |  100%  | 90.2% | python train.py                                              |
|    [ResNet-50 (all tricks+Circle)]     |  100%  | 92.2% | python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5_circle  --circle |
|     [HRNet-18 (all tricks+Circle)]     |  100%  | 92.5% | python train.py --use_hr --name  hr18_p0.5_circle_w5_b16_lr0.01_DG --lr 0.01 --batch 16  --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name  hr18_p0.5_circle_w5_b16_lr0.01_DG |
|  [SwinV2 (all tricks+Circle 256x128)]  |  100%  | 92.6% | python train.py --use_swinv2 --name swinv2_p0.5_circle_w5_b16_lr0.03  --lr 0.03 --batch 16 --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name   swinv2_p0.5_circle_w5_b16_lr0.03 --batch 32 |
| [Swin (all tricks+Circle+b16 224x224)] |  100%  | 93.1% | python train.py --use_swin --name swin_p0.5_circle_w5_b16_lr0.01 --lr 0.01 --batch 16  --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5_b16_lr0.01 |

## Getting Started

### 一、从标注视频中截取草莓图像（可跳过此步，Market为已经制作好的数据集，有兴趣的可尝试自己制作）

在Video_Strawberry_Screenshot&tools文件夹下打开Terminal输入以下命令，可实现从标注好的视频中截取草莓图像（运行前请务必修改截取后草莓图像的保存位置）

```
python Video_Strawberry_Screenshot.py
```

### 二、草莓数据集

截取后的草莓图像

```text
├── save_pic/
│   ├── 0000 
│       ├── 0000_L2_2_Unripe_1_329frame.jpg
		├── 0000_L2_2_Unripe_1_330frame.jpg
│
│   ├── 0001         
│   ├── 0002                    
│   ...  
│   ├── 1106 
```

0000~0553分配到train文件夹下

0554~1106分配到gallery文件夹下

运行以下代码的功能是实现将指定文件移动到指定文件夹下，可以帮助我们构建验证集val以及query

```python
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
```

这是最终的数据集格式

```text
├── Market/
│   ├── pytorch/
│       ├── train/                    /* 训练集train 673items 
│           ├── 0002
|           ├── 0007
|           ...
│       ├── val/                     /* 验证集val 673items（每个item中一张图像）
│       ├── train_all/               /* （训练集+验证集）train+val        
│       ├── query/                   /* query files 298items  
│       ├── gallery/                 /* gallery files 298items
```

### 三、训练

准备好数据集后，就可以开始训练了。

我们可以输入如下命令开始训练：

```
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```

`--gpu_ids` which gpu to run.

`--name` the name of the model.

`--data_dir` the path of the training data.

`--train_all` using all images to train.

`--batchsize` batch size.

`--erasing_p` random erasing probability

### 四、测试

这一部分, 我们载入我们刚刚训练的模型 来抽取每张图片的视觉特征

```
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 060
```

`--gpu_ids` which gpu to run.

`--name` the dir name of the trained model.

`--batchsize` batch size.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.

### 五、评测

是的，现在我们有了每张图片的特征。 我们需要做的事情只有用特征去匹配图像

```
python evaluate_gpu.py
```

### 六、可视化程序

测试单个query，可输入以下命令

```
python demo.py --query_index 9
```

--query_index ` which query you want to test. You may select a number in the range of `0 ~ 287

## Other tools

### 一、批量测试query

在demo.py中每次只能测试一个query，运行下面命令可以实现query的批量测试，并且将测试结果保存到指定文件夹下

```
python demo_batch.py
```

### 二、同ID下相邻帧的相似度计算

#### 1、非指数法（计算相邻帧图像的余弦距离）

```
python Calcuation_of_similarity_between_adjacent_frames.py
```

#### 2、指数法

```
python Index_method_similarity_between_adjacent_frames.py
```

### 三、数据集分析

1、图像分辨率分布图

![img](https://w6k74z8wth.feishu.cn/space/api/box/stream/download/asynccode/?code=NjQxOWQ4NmZmZmE2YzYxNTNkNThmYjRiNDc2MWRiOTBfWFp0amZxTjNrYTdkc2Z0d0RINFdVQjFOSWc1Uk52M1BfVG9rZW46WmZMemJrRWxKbzZaQmZ4bHBoWmN3TU5sbnVmXzE3MjEzNzI2OTA6MTcyMTM3NjI5MF9WNA)

```
python width_height_img.py
```

2、中心点坐标分布图

![img](https://w6k74z8wth.feishu.cn/space/api/box/stream/download/asynccode/?code=YTYyN2E2MmU3ZDRkMmNlMjdhZWU5NTYzZjA5ODdhNTlfSVNzeWRLT0luNmZzV0VTR3liUVo4WUI4aEo1U2NqNjhfVG9rZW46T3RuRGI5WFlyb3A4RU94bURJWGNVajBMbm5nXzE3MjEzNzMwNDc6MTcyMTM3NjY0N19WNA)

```
python center_coordinate.py
```

3、矩形框概览

![img](https://w6k74z8wth.feishu.cn/space/api/box/stream/download/asynccode/?code=YmVmMWEwN2RlODU4MzQ4NjRiNzc2NGExMWVhZGEyYzdfd3dvMmJBaGJpWEQ4N0FFMU5IMzN4VTlySW1NYXp0c3hfVG9rZW46R2xyMGJnYnNMb1lCR3V4TXUxZmNod1l4bkpkXzE3MjEzNzMxMzE6MTcyMTM3NjczMV9WNA)

```
python Draw_rectangular_box.py
```
