import os
import cv2
import torch
import torch.utils.data
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from skimage.feature import peak_local_max
from scipy.special import softmax as softmax_fn
from scipy.stats import entropy as entropy_fn

# EGL sampling
from utils import heatmap_loss
from utils import shannon_entropy
from utils import heatmap_generator

class ActiveLearning(object):
    '''
    Contains collection of active learning algorithms for human joint localization
    '''

    def __init__(self, conf, hg_network, learnloss_network):
        self.conf = conf
        self.hg_network = hg_network
        self.learnloss_network = learnloss_network

        self.hg_network.eval()
        self.learnloss_network.eval()

    def random(self, train, dataset_size):
        '''
        Randomly samples images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling
        '''
        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))

        # Load previously annotated images indices
        if self.conf.model_load_hg:
            annotated_idx = np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))

        num_images = self.conf.active_learning_params['num_images']

        # Determine if per dataset sampling or overall
        if self.conf.args['mpii_only']:
            overall_annotate = np.random.choice(unlabelled_idx, size=num_images['total'], replace=False)

            # Update annotated images indices
            annotated_idx = np.concatenate([annotated_idx, overall_annotate], axis=0).astype(np.int32)

        else:
            # Separation index between datasets
            accum_lspet = dataset_size['lspet']['train']
            accum_lsp = dataset_size['lspet']['train'] + dataset_size['lsp']['train']

            # Find indices which are not annotated for each dataset
            lspet_unlabelled = unlabelled_idx[np.where(np.logical_and(unlabelled_idx >= 0, unlabelled_idx < accum_lspet))]
            lsp_unlabelled = unlabelled_idx[np.where(np.logical_and(unlabelled_idx >= accum_lspet, unlabelled_idx < accum_lsp))]

            # Randomly sample images from each dataset
            lspet_annotated = np.random.choice(lspet_unlabelled, size=num_images['lspet'], replace=False)
            lsp_annotated = np.random.choice(lsp_unlabelled, size=num_images['lsp'], replace=False)

            # Update annotated images indices
            annotated_idx = np.concatenate([annotated_idx, lspet_annotated, lsp_annotated], axis=0).astype(np.int32)

        unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True)
        proportion = { key: value for (key, value) in zip(unique, counts)}
        with open(self.conf.model_save_path.format('sampling_proportion.txt'), "x") as file:
            file.write('Random sampling\n')
            [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]

        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)

        return annotated_idx


    def coreset_sampling(self, train, dataset_size):
        '''
        Coreset sampling of images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling
        '''

        def update_distances(cluster_centers, encoding, min_distances=None):
            '''
            Based on: https://github.com/google/active-learning/blob/master/sampling_methods/kcenter_greedy.py
            Update min distances given cluster centers.
            Args:
              cluster_centers: indices of cluster centers
              only_new: only calculate distance for newly selected points and update
                min_distances.
              rest_dist: whether to reset min_distances.
            '''

            if len(cluster_centers) != 0:
                # Update min_distances for all examples given new cluster center.
                x = encoding[cluster_centers]
                dist = pairwise_distances(encoding, x, metric='euclidean')

                if min_distances is None:
                    min_distances = np.min(dist, axis=1).reshape(-1, 1)
                else:
                    min_distances = np.minimum(min_distances, dist)

            return min_distances

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))

        if self.conf.model_load_hg:
            annotated_idx = np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        dataset_ = ActiveLearningDataset(train)
        coreset_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False, num_workers=8)

        hg_encoding = None

        # Part 1: Obtain embeddings
        # Disable autograd to speed up inference
        with torch.no_grad():
            for images in tqdm(coreset_dataloader):

                images = images.to(non_blocking=True, device='cuda')
                _, hourglass_features = self.hg_network(images)

                try:
                    hg_encoding = torch.cat((hg_encoding, hourglass_features['penultimate'].cpu()), dim=0)
                except TypeError:
                    hg_encoding = hourglass_features['penultimate'].cpu()

        hg_final_encoding = hg_encoding.squeeze().numpy()
        logging.info('Core-Set encodings computed.')

        # Part 2: k-Centre Greedy
        core_set_budget = self.conf.active_learning_params['num_images']['total']
        min_distances = None

        if len(annotated_idx) != 0:
            min_distances = update_distances(cluster_centers=annotated_idx, encoding=hg_final_encoding, min_distances=None)

        for _ in tqdm(range(core_set_budget)):
            if len(annotated_idx) == 0:  # Initial choice of point
                # Initialize center with a randomly selected datapoint
                ind = np.random.choice(np.arange(hg_final_encoding.shape[0]))
            else:
                ind = np.argmax(min_distances)

            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            min_distances = update_distances(cluster_centers=[ind], encoding=hg_final_encoding, min_distances=min_distances)

            annotated_idx = np.concatenate([annotated_idx, [ind]], axis=0).astype(np.int32)

        unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True)
        proportion = {key: value for (key, value) in zip(unique, counts)}
        with open(self.conf.model_save_path.format('sampling_proportion.txt'), "x") as file:
            file.write('Coreset sampling\n')
            [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]

        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)

        return annotated_idx


    def learning_loss_sampling(self, train, dataset_size, hg_depth=4):
        '''
        Learning loss sampling of images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling
        '''

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))

        if self.conf.model_load_hg:
            annotated_idx = np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        # Set of indices not annotated
        unlabelled_idx = np.array(list(set(train['index'])-set(annotated_idx)))
        unlabelled_dataset = {}

        for key in train.keys():
            unlabelled_dataset[key] = train[key][unlabelled_idx]

        dataset_ = ActiveLearningDataset(unlabelled_dataset)
        learnloss_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False, num_workers=8)

        learnloss_pred = None

        # Prediction and concatenation of the learning loss network outputs
        with torch.no_grad():
            for images in tqdm(learnloss_dataloader):

                images = images.to(non_blocking=True, device='cuda')

                _, hourglass_features = self.hg_network(images)

                if self.conf.learning_loss_original:
                    # encodings = torch.cat([hourglass_features[depth] for depth in range(1, hg_depth + 2)], dim=-1)
                    encodings = hourglass_features['penultimate']

                else:
                    # No longer concatenating, will now combine features through convolutional layers
                    encodings = torch.cat(
                        [hourglass_features['feature_5'].reshape(images.shape[0], hourglass_features['feature_5'].shape[1], -1),
                         hourglass_features['feature_4'].reshape(images.shape[0], hourglass_features['feature_4'].shape[1], -1),
                         hourglass_features['feature_3'].reshape(images.shape[0], hourglass_features['feature_3'].shape[1], -1),
                         hourglass_features['feature_2'].reshape(images.shape[0], hourglass_features['feature_2'].shape[1], -1),
                         hourglass_features['feature_1'].reshape(images.shape[0], hourglass_features['feature_1'].shape[1], -1)], dim=2)

                learnloss_pred_, _ = self.learnloss_network(encodings)
                learnloss_pred_ = learnloss_pred_.squeeze()

                try:
                    learnloss_pred = torch.cat([learnloss_pred, learnloss_pred_.cpu()], dim=0)
                except TypeError:
                    learnloss_pred = learnloss_pred_.cpu()

        # argsort defaults to ascending
        pred_with_index = np.concatenate([learnloss_pred.numpy().reshape(-1, 1),
                                          unlabelled_idx.reshape(-1, 1)], axis=-1)

        pred_with_index = pred_with_index[pred_with_index[:, 0].argsort()]
        indices = pred_with_index[-self.conf.active_learning_params['num_images']['total']:, 1]

        annotated_idx = np.concatenate([annotated_idx, indices], axis=0).astype(np.int32)

        unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True)
        proportion = {key: value for (key, value) in zip(unique, counts)}
        with open(self.conf.model_save_path.format('sampling_proportion.txt'), "x") as file:
            file.write('Learning Loss sampling\n')
            [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]

        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)

        return annotated_idx


    def expected_gradient_length_sampling(self, train, dataset_size):
        """
        https://arxiv.org/abs/2104.09493
        """
        raise NotImplementedError('The proposed Expected Gradient Length (EGL++) method is currently under review.')


    def multipeak_entropy(self, train, dataset_size):
        '''
        Multi-peak entropy sampling of images from training dataset
        :param train: (dict) Training dataset
        :param dataset_size: (dict) Stores the size of each dataset - MPII / LSP+LSPET
        :return: (np.ndarray) Indices chosen for sampling
        '''

        if self.conf.resume_training:
            return np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))

        if self.conf.model_load_hg:
            annotated_idx = np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/annotation.npy'))
        else:
            annotated_idx = np.array([])

        unlabelled_idx = np.array(list(set(train['index']) - set(annotated_idx)))

        # Multi-peak entropy only over the unlabelled set of images
        dataset_ = ActiveLearningDataset(dataset_dict=train, indices=unlabelled_idx)
        mpe_dataloader = torch.utils.data.DataLoader(dataset_, batch_size=self.conf.batch_size, shuffle=False,
                                                     num_workers=8)

        hg_heatmaps = None

        # Part 1: Obtain set of heatmaps
        # Disable autograd to speed up inference
        with torch.no_grad():
            for images in tqdm(mpe_dataloader):

                images = images.to(non_blocking=True, device='cuda')
                hg_heatmaps_, _ = self.hg_network(images)

                try:
                    hg_heatmaps = torch.cat((hg_heatmaps, hg_heatmaps_[:, -1, :, :, :].cpu()), dim=0)
                except TypeError:
                    hg_heatmaps = hg_heatmaps_[:, -1, :, :, :].cpu()

        hg_final_heatmaps = hg_heatmaps.squeeze().numpy()
        logging.info('Multi-peak entropy heatmaps computed.')


        # Part 2: Multi-peak entropy
        mpe_budget = self.conf.active_learning_params['num_images']['total']
        mpe_value_per_img = np.zeros(hg_final_heatmaps.shape[0], dtype=np.float32)

        # e.g. shape of heatmap final is BS x 14 x 64 x 64
        for i in tqdm(range(hg_final_heatmaps.shape[0])):
            normalizer = 0
            entropy = 0
            for hm in range(hg_final_heatmaps.shape[1]):
                loc = peak_local_max(hg_final_heatmaps[i, hm], min_distance=7, threshold_abs=7.5)
                peaks = hg_final_heatmaps[i, hm][loc[:, 0], loc[:, 1]]

                if peaks.shape[0] > 0:
                    normalizer += 1
                    peaks = softmax_fn(peaks)
                    entropy += entropy_fn(peaks)

            mpe_value_per_img[i] = entropy

        mpe_value_per_img = torch.from_numpy(mpe_value_per_img)
        vals, idx = torch.topk(mpe_value_per_img, k=mpe_budget, sorted=False, largest=True)
        assert idx.dim() == 1, "'idx' should be a single dimensional array"
        annotated_idx = np.concatenate([annotated_idx, unlabelled_idx[idx.numpy()]], axis=0).astype(np.int32)

        unique, counts = np.unique(train['dataset'][annotated_idx], return_counts=True)
        proportion = {key: value for (key, value) in zip(unique, counts)}
        with open(self.conf.model_save_path.format('sampling_proportion.txt'), "x") as file:
            file.write('MPE sampling\n')
            [file.write("{}: {}\n".format(key, proportion[key])) for key in proportion.keys()]

        np.save(file=self.conf.model_save_path.format('annotation.npy'), arr=annotated_idx)

        return annotated_idx


class ActiveLearningDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dict, indices=None):
        '''
        Helper class to initialize Dataset for torch Dataloader
        :param dataset_dict: (dict) Containing the dataset in numpy format
        :param indices: (np.ndarray) Which indices to use for generating the dataset
        '''
        if indices is None:
            self.images = dataset_dict['img']
            self.bounding_box = dataset_dict['bbox_coords']

        else:
            self.images = dataset_dict['img'][indices]
            self.bounding_box = dataset_dict['bbox_coords'][indices]

        self.xy_to_uv = lambda xy: (xy[1], xy[0])

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, item):
        '''

        :param item:
        :return:
        '''

        image =  self.images[item]
        bounding_box = self.bounding_box[item]

        # Determine crop
        img_shape = np.array(image.shape)

        # Bounding box for the first person
        [min_x, min_y, max_x, max_y] = bounding_box[0]

        tl_uv = self.xy_to_uv(np.array([min_x, min_y]))
        br_uv = self.xy_to_uv(np.array([max_x, max_y]))
        min_u = tl_uv[0]
        min_v = tl_uv[1]
        max_u = br_uv[0]
        max_v = br_uv[1]

        centre = np.array([(min_u + max_u) / 2, (min_v + max_v) / 2])
        height = max_u - min_u
        width = max_v - min_v

        scale = 2.0

        top_left = np.array([centre[0] - (scale * height / 2), centre[1] - (scale * width / 2)])
        bottom_right = np.array([centre[0] + (scale * height / 2), centre[1] + (scale * width / 2)])

        top_left = np.maximum(np.array([0, 0], dtype=np.int16), top_left.astype(np.int16))
        bottom_right = np.minimum(img_shape.astype(np.int16)[:-1], bottom_right.astype(np.int16))

        # Cropping the image
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]

        # Resize the image
        image = self.resize_image(image, target_size=[256, 256, 3])

        return torch.tensor(data=image / 256.0, dtype=torch.float32, device='cpu')

    def resize_image(self, image_=None, target_size=None):
        '''

        :return:
        '''
        # Compute the aspect ratios
        image_aspect_ratio = image_.shape[0] / image_.shape[1]
        tgt_aspect_ratio = target_size[0] / target_size[1]

        # Compare the original and target aspect ratio
        if image_aspect_ratio > tgt_aspect_ratio:
            # If target aspect ratio is smaller, scale the first dim
            scale_factor = target_size[0] / image_.shape[0]
        else:
            # If target aspect ratio is bigger or equal, scale the second dim
            scale_factor = target_size[1] / image_.shape[1]

        # Compute the padding to fit the target size
        pad_u = (target_size[0] - int(image_.shape[0] * scale_factor))
        pad_v = (target_size[1] - int(image_.shape[1] * scale_factor))

        output_img = np.zeros(target_size, dtype=image_.dtype)

        # Write scaled size in reverse order because opencv resize
        scaled_size = (int(image_.shape[1] * scale_factor), int(image_.shape[0] * scale_factor))

        padding_u = int(pad_u / 2)
        padding_v = int(pad_v / 2)

        im_scaled = cv2.resize(image_, scaled_size)
        # logging.debug('Scaled, pre-padding size: {}'.format(im_scaled.shape))

        output_img[padding_u : im_scaled.shape[0] + padding_u,
                   padding_v : im_scaled.shape[1] + padding_v, :] = im_scaled

        return output_img
