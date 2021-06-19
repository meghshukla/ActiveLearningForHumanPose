import os
import sys
import copy
import math
from pathlib import Path

import torch
import scipy.io
import numpy as np
from tqdm import tqdm
from adjustText import adjust_text
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle

import umap
from sklearn.decomposition import PCA

plt.switch_backend('agg')


def visualize_image(image_info, bbox=False, uv=False):
    '''
    :param image_info: (dict)
    '''

    uv_to_xy = lambda uv: (uv[1], uv[0])

    root = Path(os.getcwd()).parent
    sys.path.append(root)

    colour = {'rankl': (0, 0, 1), 'rknee': (0, 0, 1), 'rhip': (0, 0, 1),
              'lankl': (1, 0, 0), 'lknee': (1, 0, 0), 'lhip': (1, 0, 0),
              'rwri': (1, 1, 0), 'relb': (1, 1, 0), 'rsho': (1, 1, 0),
              'lwri': (0, 1, 0), 'lelb': (0, 1, 0), 'lsho': (0, 1, 0),
              'head': (0, 1, 1), 'neck': (0, 1, 1)}

    os.makedirs(os.path.join(root, 'results', 'viz_gt'), exist_ok=True)
    img_dump = os.path.join(root, 'results', 'viz_gt')

    # Currently will iterate over MPII and LSPET and LSP
    for dataset_name_ in image_info.keys():
        # Iterate over all images
        for i in tqdm(range(len(image_info[dataset_name_]['img']))):

            fig, ax = plt.subplots(nrows=1, ncols=1, frameon=False)
            ax.set_axis_off()

            img = image_info[dataset_name_]['img'][i]
            img_name = image_info[dataset_name_]['img_name'][i]
            img_pred = image_info[dataset_name_]['img_pred'][i]
            img_gt = image_info[dataset_name_]['img_gt'][i]
            img_split = image_info[dataset_name_]['split'][i]
            img_dataset = image_info[dataset_name_]['dataset'][i]

            # One list for each, ground truth and predictions
            text_overlay = []
            ax.set_title('Name: {}, Shape: {}, Split: {}, Dataset: {}'.format(img_name, str(img.shape), img_split, img_dataset),
                         color='white')
            ax.imshow(img)

            joint_names = list(colour.keys())
            for jnt in joint_names:
                for jnt_gt in img_gt[jnt]:
                    if jnt_gt[2] == 1:
                        if uv:
                            jnt_gt = uv_to_xy(jnt_gt)
                        text_overlay.append(ax.text(x=jnt_gt[0], y=jnt_gt[1], s=jnt, color=colour[jnt], fontsize=6))
                        ax.add_patch(Circle(jnt_gt[:2], radius=2.5, color=colour[jnt], fill=False))

            for jnt in joint_names:
                for jnt_pred in img_pred[jnt]:
                    if jnt_pred[2] == 1:
                        if uv:
                            jnt_pred = uv_to_xy(jnt_pred)
                        text_overlay.append(ax.text(x=jnt_pred[0], y=jnt_pred[1], s=jnt, color=colour[jnt], fontsize=6))
                        ax.add_patch(Circle(jnt_pred[:2], radius=2.5, color=colour[jnt], fill=True))

            if bbox:
                for person_patch in range(image_info[dataset_name_]['bbox_coords'].shape[1]):
                    coords = image_info[dataset_name_]['bbox_coords'][i, person_patch]
                    ax.add_patch(Rectangle(xy=(coords[0], coords[1]), height=(coords[3] - coords[1]), width=(coords[2]-coords[0]),
                                           linewidth=1, edgecolor='r', fill=False))

            adjust_text(text_overlay)

            plt.savefig(fname=os.path.join(img_dump, '{}'.format(img_name)),
                        facecolor='black', edgecolor='black', bbox_inches='tight', dpi=300)

            del fig, ax
            plt.close()


def heatmap_loss(combined_hm_preds, heatmaps, nstack, egl=False):
    '''

    :param combined_hm_preds:
    :param heatmaps:
    :param nstack:
    :return:
    '''

    combined_loss = []
    calc_loss = lambda pred, gt:  ((pred - gt)**2).mean(dim=[1, 2, 3])

    for i in range(nstack):
        if egl:
            combined_loss.append(calc_loss(combined_hm_preds[:, i], heatmaps[:, i]))
        else:
            combined_loss.append(calc_loss(combined_hm_preds[:, i], heatmaps))
    combined_loss = torch.stack(combined_loss, dim=1)

    return combined_loss


def heatmap_generator(joints, occlusion, hm_shape=(0, 0), img_shape=(0, 0)):
    '''

    :param joints:
    :return:
    '''

    def draw_heatmap(pt_uv, sigma=1.75, use_occlusion=False, hm_shape=(0, 0)):
        '''
        2D gaussian (exponential term only) centred at given point.
        No constraints on point to be integer only.
        :param im: (Numpy array of size=64x64) Heatmap
        :param pt: (Numpy array of size=2) Float values denoting point on the heatmap
        :param sigma: (Float) self.joint_size which determines the standard deviation of the gaussian
        :return: (Numpy array of size=64x64) Heatmap with gaussian centred around point.
        '''

        im = np.zeros(hm_shape, dtype=np.float32)

        # If coordinates are negative OR coordinates affined beyond visibility OR joint is absent
        if (pt_uv[2] == -1) or (pt_uv[0] < 0) or (pt_uv[1] < 0) or (pt_uv[0] > hm_shape[0]) or (pt_uv[1] > hm_shape[1]):
            return im, 0

        elif pt_uv[2] == 0:
            if not use_occlusion:
                return im, 0
            pt_uv = pt_uv[:2]

        else:
            assert pt_uv[2] == 1, "joint[2] should be (-1, 0, 1), but got {}".format(pt_uv[2])
            pt_uv = pt_uv[:2]

        # Point around which Gaussian will be centred.
        pt_uv_rint = np.rint(pt_uv).astype(int)

        # Size of 2D Gaussian window.
        size = int(math.ceil(6 * sigma))
        # Ensuring that size remains an odd number
        if not size % 2:
            size += 1

        # Generate gaussian, with window=size and variance=sigma
        u = np.arange(pt_uv_rint[0] - (size // 2), pt_uv_rint[0] + (size // 2) + 1)
        v = np.arange(pt_uv_rint[1] - (size // 2), pt_uv_rint[1] + (size // 2) + 1)
        uu, vv = np.meshgrid(u, v, sparse=True)
        z = np.exp(-((uu - pt_uv[0]) ** 2 + (vv - pt_uv[1]) ** 2) / (2 * (sigma ** 2)))
        z = z.T

        # Crop Size for upper left and bottom right coordinates
        ul_u = min(pt_uv_rint[0], size // 2)
        ul_v = min(pt_uv_rint[1], size // 2)
        br_u = min((im.shape[0] - 1) - pt_uv_rint[0], size // 2)
        br_v = min((im.shape[1] - 1) - pt_uv_rint[1], size // 2)

        # Crop around the centre of the gaussian.
        gauss_crop = z[(size // 2) - ul_u:(size // 2) + br_u + 1,
                     (size // 2) - ul_v:(size // 2) + br_v + 1]

        # Heatmap = crop
        im[pt_uv_rint[0] - ul_u:pt_uv_rint[0] + br_u + 1,
        pt_uv_rint[1] - ul_v:pt_uv_rint[1] + br_v + 1] = gauss_crop

        return im, 1   # heatmap, joint_exist

    assert len(joints.shape) == 3, 'Joints should be rank 3:' \
                                   '(num_person, num_joints, [u,v,vis]), but is instead {}'.format(joints.shape)

    heatmaps = np.zeros([joints.shape[1], hm_shape[0], hm_shape[1]], dtype=np.float32)
    joints_exist = np.zeros([joints.shape[1]], dtype=np.uint8)

    # Downscale
    downscale = [(img_shape[0] - 1)/(hm_shape[0] - 1), ((img_shape[1] - 1)/(hm_shape[1] - 1))]
    joints /= np.array([downscale[0], downscale[1], 1]).reshape(1, 1, 3)

    # Iterate over number of heatmaps
    for i in range(joints.shape[1]):

        # Create new heatmap for joint
        hm_i = np.zeros(hm_shape, dtype=np.float32)

        # Iterate over persons
        for p in range(joints.shape[0]):
            hm_, joint_present = draw_heatmap(pt_uv=joints[p, i, :], use_occlusion=occlusion, hm_shape=hm_shape)
            joints_exist[i] = max(joints_exist[i], joint_present)
            hm_i = np.maximum(hm_i, hm_)

        heatmaps[i] = hm_i

    return heatmaps, joints_exist


def uv_from_heatmap(hm=None, threshold=None, img_shape=(256, 256)):
    '''

    :param hm:
    :param threshold:
    :param img_shape:
    :return:
    '''
    max_uv = arg_max(hm)
    corrected_uv = weight_avg_centre(hm=hm, max_uv=max_uv)

    if hm[int(corrected_uv[0]), int(corrected_uv[1])] < threshold:
        return np.array([-1, -1, -1])

    else:
        joints = np.array([corrected_uv[0], corrected_uv[1], 1])
        hm_shape = hm.shape
        upscale = [(img_shape[0] - 1) / (hm_shape[0] - 1), ((img_shape[1] - 1) / (hm_shape[1] - 1))]
        joints *= np.array([upscale[0], upscale[1], 1])

        return joints


def arg_max(img):
    '''
    Find the indices corresponding to maximum values in the heatmap
    :param img: (Numpy array of size=64x64) Heatmap
    :return: (Torch tensor of size=2) argmax of the image
    '''
    img = torch.tensor(img)
    assert img.dim() == 2, 'Expected img.dim() == 2, got {}'.format(img.dim())

    h = img.shape[0]
    w = img.shape[1]

    rawmaxidx = img.flatten().argmax()

    max_u = int(rawmaxidx) // int(w)
    max_v = int(rawmaxidx) % int(w)

    return torch.FloatTensor([max_u, max_v])


def weight_avg_centre(hm, max_uv=None, jnt_size=1.75):
    '''
    Weighted average of points around the maxima. Weighted average avoids solitary spikes being identified.
    :param hm: (Numpy array of size 64x64)
    :param jnt_size: (Float) Windows size around the maxima to compute weighted average.
    :return: (Numpy array of size=2)
    '''

    hm = torch.clamp(torch.from_numpy(hm), min=0.0)
    mx = max_uv

    # Dimension of the heatmap
    siz = torch.Tensor([hm.shape[0], hm.shape[1]]).float()

    # Clip indices if needed so that start and end indices are valid points.
    st_idx = torch.max(torch.zeros(2), mx - np.ceil(jnt_size))
    end_idx = torch.min(siz - 1, mx + np.ceil(jnt_size))

    # Crop around the maxima.
    img_crop = hm[int(st_idx[0]):int(end_idx[0] + 1), int(st_idx[1]):int(end_idx[1] + 1)].clone()
    img_crop = img_crop.type(torch.FloatTensor)
    img_sum = img_crop.sum()
    if img_sum == 0:
        img_sum = img_sum + 0.000001

    # Weighted average along column/row
    u = img_crop.sum(1).mul(torch.arange(st_idx[0], end_idx[0] + 1)).div(img_sum).sum()
    v = img_crop.sum(0).mul(torch.arange(st_idx[1], end_idx[1] + 1)).div(img_sum).sum()

    return np.array([u, v])


def principal_component_analysis(encodings, n_components=2):
    '''
    Compute the principal component transform of the encodings
    :param encodings: Encodings generated by LLAL network
    :param n_components: Number of components to retain
    :return: Principal Component Transform of encodings
    '''
    pca = PCA(n_components=n_components)
    pca.fit(encodings)
    pca_encodings = pca.transform(encodings)

    return pca_encodings


def umap_fn(encodings, n_components=2):
    '''
    NUMPY
    https://umap-learn.readthedocs.io/en/latest/how_umap_works.html
    Compute the UMAP transform of the encodings
    :param encodings: Encodings generated by LLAL network
    :param n_components: Number of components to retain
    :return: UMAP transform of the encodings
    '''
    # Number of neighbours balances the local versus the global structure of the data
    umap_transform = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=n_components).fit(encodings)
    umap_encodings = umap_transform.transform(encodings)

    return umap_encodings


def shannon_entropy(probs):
    '''
    Computes the Shannon Entropy for a distribution
    :param probs_array: 2D-Tensor; Probability distribution along axis=1
    :return: Scalar; H(p)
    '''
    return torch.sum(-probs * torch.log(probs), dim=1)