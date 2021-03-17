
import cv2
from PIL import Image
from matplotlib import cm
import torchsummary
import imageio
from pathlib import Path
import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18, resnet34
import datasets.mvtec as mvtec


# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
resize = 512


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2', 'resnet34', 'resnet50'], default='resnet34')
    return parser.parse_args()


def main():

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=True, progress=True)
        # torchsummary.summary(model, (3, 512, 512), device='cpu')
        t_d = 1792
        d = 550
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=True, progress=True)
        t_d = 256
        d = 100

    model.to(device)
    model.eval()

    # random.seed(1024)
    # torch.manual_seed(1024)
    # if use_cuda:
    #     torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d))

    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    # files = Path('/tmp/garbage/').glob('*.jpg')
    # folder = '/home/daniele/Desktop/experiments/2021-01-28.PlaygroundDatasets/lego_00/data/'

    for j in range(100):
        preds = []
        for i in [1, 2, 3, 4, 5, 6]:
            # for i in [7, 8, 9, 10, 11, 12]:
            tmp = '/home/daniele/Desktop/experiments/2021-01-28.PlaygroundDatasets/lego_00/data/{}_image.jpg'

            x = load_image(tmp.format(str(j * 13 + i).zfill(5)))
            embedding_vectors = compute_embeddings(x, model, outputs)
            print(embedding_vectors.shape)
            preds.append(embedding_vectors)

        preds = torch.cat(preds)
        print(preds.shape)
        B, C, H, W = preds.size()
        preds = preds.view(B, C, H * W)
        print("1", preds.shape)
        v = torch.var(preds, dim=0)
        print("2", v.shape)
        v = v.view(C, H, W).mean(dim=0)
        print(v.shape)
        # diff = torch.nn.functional.l1_loss(embedding_vectors_0, embedding_vectors_1, reduction='none').mean(dim=1).squeeze(0)

        diff = v.detach().cpu().numpy()
        print(v.min(), v.max())
        diff = (diff - diff.min()) / (diff.max() - diff.min())
        diff = cv2.resize(diff, (resize, resize))
        cv2.namedWindow("diff", cv2.WINDOW_NORMAL)
        cv2.imshow("diff", diff)
        if ord('q') == cv2.waitKey(0):
            import sys
            sys.exit(0)


def load_image(path):
    image = Image.open(path)
    image = image.resize([resize, resize])
    image = np.array(image) / 255.
    x = torch.Tensor(image).permute(2, 0, 1).unsqueeze(0)
    return x


def compute_embeddings(x, model, outputs):
    outputs.clear()
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    with torch.no_grad():
        _ = model(x.to(device))
    # get intermediate layer outputs
    for k, v in zip(train_outputs.keys(), outputs):
        train_outputs[k].append(v.cpu().detach())

    for k, v in train_outputs.items():
        train_outputs[k] = torch.cat(v, 0)

    embedding_vectors = train_outputs['layer1']
    # for layer_name in ['layer2', 'layer3']:
    #     embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])
    return embedding_vectors


def apply_colormap(img, cmap='magma', normed: bool = False):
    """ Applies colormap to image [HxWx3] as uint8

    :param img: image [HxWx3] as uint8
    :type img: np.ndarray
    :param cmap: matplotlib colormap, defaults to 'magma'
    :type cmap: str, optional
    """

    img = img.astype(float) / 255.
    my_cm = cm.get_cmap('magma')
    if normed:
        normed_data = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        normed_data = img
    img = my_cm(normed_data)
    img = (img * 255.).astype(np.uint8)
    import cv2
    img = cv2.medianBlur(img, 25)
    return img


def plot_fig(test_img, scores, gts, threshold, save_dir, class_name):
    num = len(scores)
    vmax = scores.max() * 255.
    vmin = scores.min() * 255.
    for i in range(num):
        img = test_img[i]
        img = denormalization(img)
        gt = gts[i].transpose(1, 2, 0).squeeze()
        heat_map = scores[i] * 255
        mask = scores[i]
        mask[mask > threshold] = 1
        mask[mask <= threshold] = 0
        kernel = morphology.disk(4)
        mask = morphology.opening(mask, kernel)
        mask *= 255
        vis_img = mark_boundaries(img, mask, color=(1, 0, 0), mode='thick')
        fig_img, ax_img = plt.subplots(1, 5, figsize=(12, 3))
        fig_img.subplots_adjust(right=0.9)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        for ax_i in ax_img:
            ax_i.axes.xaxis.set_visible(False)
            ax_i.axes.yaxis.set_visible(False)

        out_img = Path(save_dir) / f'{str(i).zfill(5)}_image.png'
        out_heat = Path(save_dir) / f'{str(i).zfill(5)}_heatmap.png'
        out_heatc = Path(save_dir) / f'{str(i).zfill(5)}_heatmapc.png'
        out_mask = Path(save_dir) / f'{str(i).zfill(5)}_mask.png'
        out_vis = Path(save_dir) / f'{str(i).zfill(5)}_vis.png'

        heat_mapc = apply_colormap(heat_map)

        [imageio.imwrite(x, y) for x, y in zip(
            [out_img, out_heat, out_heatc, out_mask, out_vis],
            [img, heat_map, heat_mapc, mask * 255, vis_img]
        )]

        print("OUT", img.shape, heat_map.shape, mask.shape)
        ax_img[0].imshow(img)
        ax_img[0].title.set_text('Image')
        ax_img[1].imshow(gt, cmap='gray')
        ax_img[1].title.set_text('GroundTruth')
        ax = ax_img[2].imshow(heat_map, cmap='jet', norm=norm)
        ax_img[2].imshow(img, cmap='gray', interpolation='none')
        ax_img[2].imshow(heat_map, cmap='jet', alpha=0.5, interpolation='none')
        ax_img[2].title.set_text('Predicted heat map')
        ax_img[3].imshow(mask, cmap='gray')
        ax_img[3].title.set_text('Predicted mask')
        ax_img[4].imshow(vis_img)
        ax_img[4].title.set_text('Segmentation result')
        left = 0.92
        bottom = 0.15
        width = 0.015
        height = 1 - 2 * bottom
        rect = [left, bottom, width, height]
        cbar_ax = fig_img.add_axes(rect)
        cb = plt.colorbar(ax, shrink=0.6, cax=cbar_ax, fraction=0.046)
        cb.ax.tick_params(labelsize=8)
        font = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
            'size': 8,
        }
        cb.set_label('Anomaly Score', fontdict=font)

        # fig_img.savefig(os.path.join(save_dir, class_name + '_{}'.format(i)), dpi=100)
        plt.close()


def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)

    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
