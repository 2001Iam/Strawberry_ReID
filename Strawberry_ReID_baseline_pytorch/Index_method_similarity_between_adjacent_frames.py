# 计算同一个草莓ID下相邻帧相似度，并将得到的相邻帧相似度保存到指定文件夹下（一个ID一张图像，图像中包含该ID下所有相邻帧相似度)
# score的计算方法同Calcuation_of_similarity_between_adjacent_frames.py不同 ---get_score2()
# 每一个帧与相邻相似度的计算不仅用到了相邻帧的特征向量信息，还用到了历史的特征向量的信息
# h0=0
# h1=0.8*F1+0.2h0
# score1=cos(h1,F2)
# h2=0.8*F2+0.2h1
# score2=cos(h2,F3)
import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

k = 1
fig = plt.figure(figsize=(16, 16))


def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def get_score1(qf, gf):
    gallery1 = qf.view(-1, 1)
    gallery2 = gf.view(1, -1)
    score = torch.mm(gallery2, gallery1)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score


def get_score2(hf, qf, gf):
    res = hf * 0.2 + qf * 0.8
    gallery1 = res.view(-1, 1)
    gallery2 = gf.view(1, -1)
    score = torch.mm(gallery2, gallery1)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score, res #res is 历史的特征向量


flag = 1
history = []
for j in range(200):
    parser = argparse.ArgumentParser(description='Demo')
    parser.add_argument('--gallery1_index', default=j, type=int, help='test_image_index')
    parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
    parser.add_argument('--save_path', default='/home/xplv/fenghao/Index_Adjacent_frame_similarity', type=str, help='save path')
    opts = parser.parse_args()

    data_dir = opts.test_dir
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['gallery']}

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
    if flag == 1:
        flag = 0
        history = gallery_feature[i]
        score = get_score1(gallery_feature[i], gallery_feature[i + 1])

    else:
        score, history = get_score2(history, gallery_feature[i], gallery_feature[i + 1])

    ########################################################################
    # Visualize the rank result
    print(f'score:{score}')
    gallery1_path, _ = image_datasets['gallery'].imgs[i]
    gallery1_label = gallery_label[i]
    gallery1_ids = gallery1_path.split('_')
    gallery1_frames = gallery1_ids[5]
    gallery1_frame = gallery1_frames[0:4]

    gallery2_path, _ = image_datasets['gallery'].imgs[i + 1]
    gallery2_label = gallery_label[i + 1]
    gallery2_ids = gallery2_path.split('_')
    gallery2_frames = gallery2_ids[5]
    gallery2_frame = gallery2_frames[0:4]

    if gallery1_label == gallery2_label:

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

    else:
        k = 1
        flag = 1
        save_path = f'{opts.save_path}/{gallery1_label:04d}.png'
        fig.savefig(save_path)
        fig = plt.figure(figsize=(16, 16))
