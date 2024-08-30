import argparse
import scipy.io
import torch
import numpy as np
import os
from torchvision import datasets
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

#######################################################################
# Evaluate

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--query_index', default=True, type=int, help='test_image_index')
parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
parser.add_argument('--save_path', default='/home/xplv/fenghao/ResNet50_plt_savefig', type=str, help='save path')

opts = parser.parse_args()

data_dir = opts.test_dir
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x)) for x in ['gallery', 'query']}


#####################################################################
# Show result
def imshow(path, title=None):
    """Imshow for Tensor."""
    im = plt.imread(path)
    plt.imshow(im)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


######################################################################
result = scipy.io.loadmat('pytorch_result.mat')
query_feature = torch.FloatTensor(result['query_f'])
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
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

query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()


#######################################################################
# sort the images
# sort_img()函数用于将gallery中的图像与query的相似度按照从大到小的顺序进行排序
def sort_img(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    # predict index
    index = np.argsort(score)  # from small to large
    index = index[::-1]
    # index = index[0:2000]
    # good index
    query_index = np.argwhere(gl == ql)
    # same camera
    camera_index = np.argwhere(gc == qc)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]
    return index


def get_score(qf, ql, qc, gf, gl, gc):
    query = qf.view(-1, 1)
    # print(query.shape)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    return score

# count_file_in_folder()函数用于统计指定文件夹下文件的数量
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


i = opts.query_index

index = sort_img(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
score = get_score(query_feature[i], query_label[i], query_cam[i], gallery_feature, gallery_label, gallery_cam)
########################################################################
# Visualize the rank result

query_path, _ = image_datasets['query'].imgs[i]
query_label = query_label[i]
query_ids = query_path.split('_')
query_frames = query_ids[5]
query_frame = query_frames[0:4]
# 使用方法
folder_to_check = f'/home/xplv/fenghao/Market/pytorch/gallery/{query_label:04d}'
num_of_files = count_files_in_folder(folder_to_check)
print(query_path)
try:  # Visualize Ranking Result
    # Graphical User Interface is needed
    fig = plt.figure(figsize=(30, 30))
    ax = plt.subplot(6, 10, 1)
    ax.axis('off')
    imshow(query_path, f'{query_frame}frame-query_total:{num_of_files}')
    for i in range(50):
        ax = plt.subplot(6, 10, i + 11)
        ax.axis('off')
        img_path, _ = image_datasets['gallery'].imgs[index[i]]
        label = gallery_label[index[i]]
        imshow(img_path)
        str_ids = img_path.split('_')
        frame_all = str_ids[5]
        frame = frame_all[0:4]
        if label == query_label:
            # ax.set_title('%d'%(i+1), color='green')
            ax.set_title(f'{i + 1}:{score[index[i]]:.2f}_{label}_{frame}frame', color='green')
        else:
            # ax.set_title('%d'%(i+1), color='red')
            ax.set_title(f'{i + 1}:{score[index[i]]:.2f}_{label}_{frame}frame', color='red')
        print(f'{i}{img_path}')
except RuntimeError:
    for i in range(50):
        img_path = image_datasets.imgs[index[i]]
        print(img_path[0])
    print('If you want to see the visualization of the ranking result, graphical user interface is needed.')
save_path = f'{opts.save_path}/{query_label:04d}.png'
fig.savefig(save_path)
