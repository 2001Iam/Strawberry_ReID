# 计算同一个草莓ID下相邻帧相似度，并将得到的相邻帧相似度保存到指定文件夹下（一个ID一张图像，图像中包含该ID下所有相邻帧相似度)
# 该python文件是根据demo.py文件修改的，所以在该python文件中有些出现的参数并未用处
import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

k = 1  # k作为plt.subplot(5, 10, k)的参数，是一个索引，用于指定想要放置图形的位置，k初始化为1，表示从第一个位置开始放
fig = plt.figure(figsize=(16, 16))  # 生成一个新的图形容器,(16, 16)意味着创建一个16英寸乘以16英寸的图像


def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# get_score()函数用于计算输入的两张图像的相似度（即余弦距离)
# 此处的score跟demo.py中的不一样，此处输入的参数gf是单个图像的feature，且返回的score是一个实数
def get_score(qf, gf):
    gallery1 = qf.view(-1, 1)
    gallery2 = gf.view(1, -1)
    print(gallery2.shape)
    score = torch.mm(gallery2, gallery1)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score


# get_number()函数用于获取文件名中的草莓ID和帧号,最终目的是想按照ID和帧号对文件进行排序
def get_number(str):  # str格式：../Market/pytorch/gallery/0554/0554_L2_2_Unripe_721_3725frame.jpg
    num1 = int(str.split('/')[4])  # num1=554
    num2 = str.split('/')[5]
    num2 = num2.split('_')[5]
    num2 = int(num2[0:4])  # num2=3725
    return num1, num2  # num1 is ID,num2 is frame


store_path = []
image_datasets = {x: datasets.ImageFolder(os.path.join('../Market/pytorch', x)) for x in ['gallery']}  # 具体是啥可以打印出来看看
# print(len(image_datasets['gallery'].imgs))
# print(image_datasets['gallery'].imgs[0])
len_gallery = len(image_datasets['gallery'].imgs)
for n in range(0, len_gallery):  # gallery中共有6918张图像，根据文件名对6918张图像进行一个排序
    path, _ = image_datasets['gallery'].imgs[n]
    store_path.append(path)
    print(path)
sorted_path = sorted(store_path, key=get_number)  # 先按照ID排序，ID相同再按照frame排序
for j in range(0, 100):
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--gallery1_index', default=j, type=int, help='test_image_index')
    parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
    parser.add_argument('--save_path', default='/home/xplv/fenghao/ReID_project/modify_ResNet50_Adjacent_frame_similarity', type=str, help='save path')
    opts = parser.parse_args()

    # data_dir = opts.test_dir
    result = scipy.io.loadmat('pytorch_result.mat')
    gallery_feature = torch.FloatTensor(result['gallery_f'])
    gallery_cam = result['gallery_cam'][0]
    gallery_label = result['gallery_label'][0]

    multi = os.path.isfile('multi_query.mat')

    if multi:
        m_result = scipy.io.loadmat('multi_query.mat')
        mquery_feature = torch.FloatTensor(m_result['mquery_f'])
        mquery_cam = m_result['mquery_cam'][0]
        mquery_label = m_result['mquery_label'][0]
        mquery_feature = mquery_feature.cuda()
        print('multi is true')

    gallery_feature = gallery_feature.cuda()

    i = opts.gallery1_index
    score = get_score(gallery_feature[i], gallery_feature[i + 1])  # 计算相邻帧相似度
    # Visualize the rank result
    print(f'score:{score}')
    # gallery1_path, _ = image_datasets['gallery'].imgs[i]
    gallery1_path = sorted_path[i]
    gallery1_label = gallery_label[i]
    gallery1_ids = gallery1_path.split('_')
    gallery1_frames = gallery1_ids[5]
    gallery1_frame = gallery1_frames[0:4]

    # gallery2_path, _ = image_datasets['gallery'].imgs[i + 1]
    gallery2_path = sorted_path[i + 1]
    gallery2_label = gallery_label[i + 1]
    gallery2_ids = gallery2_path.split('_')
    gallery2_frames = gallery2_ids[5]
    gallery2_frame = gallery2_frames[0:4]

    if gallery1_label == gallery2_label:  # 如果gallery1和gallery2这两个图像的ID是一样的

        ax = plt.subplot(5, 10, k)
        ax.axis('off')
        imshow(gallery1_path)
        ax.set_title(f'{gallery1_frame}f_{gallery1_label}_s:{score[0]:.2f}', fontsize=8)

        ax = plt.subplot(5, 10, k + 1)
        ax.axis('off')
        imshow(gallery2_path)
        ax.set_title(f'{gallery2_frame}f_{gallery2_label}', fontsize=8)
        print(k)
        k += 1

    else:  # 如果gallery1和gallery2这两个图像的ID是不一样的，则说明gallery1是上一个ID 的最后一帧，而gallery2是新ID的第一帧
        k = 1  # 重新初始化k=1
        save_path = f'{opts.save_path}/{gallery1_label:04d}.png'
        fig.savefig(save_path)  # 保存gallery1所在ID的图像相邻帧相似度
        fig = plt.figure(figsize=(16, 16))  # 且新生成一个图形容器,用于存放gallery2所在ID的图像相邻帧相似度
