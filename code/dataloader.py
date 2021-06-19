import os
import sys
import copy
import logging
from pathlib import Path

import cv2
import scipy.io
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.patches import Circle

import torch
import torch.utils.data
import albumentations as albu

from utils import heatmap_generator
from utils import uv_from_heatmap


jnt_to_ind = {'head': 0, 'neck': 1, 'lsho': 2, 'lelb': 3, 'lwri': 4, 'rsho': 5, 'relb': 6, 'rwri': 7,
              'lhip': 8, 'lknee': 9, 'lankl': 10, 'rhip': 11, 'rknee': 12, 'rankl': 13}

ind_to_jnt = {0: 'head', 1: 'neck', 2: 'lsho', 3: 'lelb', 4: 'lwri', 5: 'rsho', 6: 'relb', 7: 'rwri',
              8: 'lhip', 9: 'lknee', 10: 'lankl', 11: 'rhip', 12: 'rknee', 13: 'rankl'}


def load_mpii(lambda_head=1.0, del_extra_jnts=False, precached_mpii=False, mpii_only=False):
    '''
    Converts Matlab structure .mat file into a more intuitive dictionary object.
    :return: (dict, int) 1. Dictionary containing the MPII dataset 2. Maximum number of people in an image
    '''

    root = Path(os.getcwd()).parent
    if not del_extra_jnts:
        global jnt_to_ind, ind_to_jnt
        jnt_to_ind['pelvis'] = 14
        jnt_to_ind['thorax'] = 15

        ind_to_jnt[14] = 'pelvis'
        ind_to_jnt[15] = 'thorax'

    if precached_mpii:
        if mpii_only:
            string = 'mpii_only'
        else:
            string = 'lsp_lspet'

        img_dict = np.load(os.path.join(root, 'cached', 'mpii_cache_{}.npy'.format(string)), allow_pickle=True)
        img_dict = img_dict[()]
        max_person_in_img = np.load(os.path.join(root, 'cached', 'mpii_maxPpl_cache_{}.npy'.format(string)), allow_pickle=True)

        try:
            assert img_dict['mpii']['del_extra_jnts'] == del_extra_jnts
            assert img_dict['mpii']['lambda_head'] == lambda_head

            del img_dict['mpii']['del_extra_jnts']
            del img_dict['mpii']['lambda_head']

            return img_dict, max_person_in_img

        except AssertionError:
            logging.warning('Cannot load MPII due to different configurations.')
            logging.warning('Loading MPII from scratch.\n')

    mpii_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                       6: 'pelvis', 7: 'thorax', 8: 'neck', 11: 'relb', 10: 'rwri', 9: 'head',
                       12: 'rsho', 13: 'lsho', 14: 'lelb', 15: 'lwri'}

    max_person_in_img = 0

    # Create a template for GT and Pred to follow
    mpii_template = dict([(mpii_idx_to_jnt[i], []) for i in range(16)])
    img_dict = {'mpii': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'normalizer': [],
                         'dataset': [], 'num_gt': [], 'split': [], 'scale': [], 'objpos': [], 'num_ppl': []}}

    # Load MPII
    matlab_mpii = scipy.io.loadmat(os.path.join(dataset_path[1], 'joints.mat'), struct_as_record=False)['RELEASE'][0, 0]

    # Iterate over all images
    # matlab_mpii.__dict__['annolist'][0].shape[0]
    for img_idx in tqdm(range(matlab_mpii.__dict__['annolist'][0].shape[0])):

        # Load annotation data per image
        annotation_mpii = matlab_mpii.__dict__['annolist'][0, img_idx]
        train_test_mpii = matlab_mpii.__dict__['img_train'][0, img_idx].flatten()[0]

        person_id = matlab_mpii.__dict__['single_person'][img_idx][0].flatten()
        num_people = len(person_id)
        max_person_in_img = max(max_person_in_img, len(person_id))

        # Read image
        img_name = annotation_mpii.__dict__['image'][0, 0].__dict__['name'][0]

        try:
            image = plt.imread(os.path.join(dataset_path[1], 'images', img_name))
        except FileNotFoundError:
            logging.warning('Could not load filename: {}'.format(img_name))
            continue

        # Create a deepcopy of the template to avoid overwriting the original
        gt_per_image = copy.deepcopy(mpii_template)
        num_joints_persons = []
        normalizer_persons = []
        scale = []
        objpos = []

        # Default is that there are no annotated people in the image
        annotated_person_flag = False

        # Iterate over each person
        for person in (person_id - 1):
            try:
                per_person_jnts = []

                # If annopoints not present, then annotations for that person absent. Throw exception and skip to next
                annopoints_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0]
                scale_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['scale'][0][0]
                objpose_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['objpos'][0][0]
                objpose_img_mpii = [objpose_img_mpii.__dict__['x'][0][0], objpose_img_mpii.__dict__['y'][0][0]]
                num_joints = annopoints_img_mpii.__dict__['point'][0].shape[0]
                remove_pelvis_thorax_from_num_joints = 0

                # PCKh@0.x: Head bounding box normalizer
                head_x1 = annotation_mpii.__dict__['annorect'][0, person].__dict__['x1'][0][0]
                head_y1 = annotation_mpii.__dict__['annorect'][0, person].__dict__['y1'][0][0]
                head_x2 = annotation_mpii.__dict__['annorect'][0, person].__dict__['x2'][0][0]
                head_y2 = annotation_mpii.__dict__['annorect'][0, person].__dict__['y2'][0][0]
                xy_1 = np.array([head_x1, head_y1], dtype=np.float32)
                xy_2 = np.array([head_x2, head_y2], dtype=np.float32)

                normalizer_persons.append(np.linalg.norm(xy_1 - xy_2, ord=2))

                # If both are true, pulls the head joint closer to the neck, and body
                head_jt, neck_jt = False, False

                # MPII does not have a [-1, -1] or absent GT, hence the number of gt differ for each image
                for i in range(num_joints):
                    x = annopoints_img_mpii.__dict__['point'][0, i].__dict__['x'].flatten()[0]
                    y = annopoints_img_mpii.__dict__['point'][0, i].__dict__['y'].flatten()[0]
                    id_ = annopoints_img_mpii.__dict__['point'][0, i].__dict__['id'][0][0]
                    vis = annopoints_img_mpii.__dict__['point'][0, i].__dict__['is_visible'].flatten()

                    # No entry corresponding to visible, mostly head vis is missing.
                    if vis.size == 0:
                        vis = 1
                    else:
                        vis = vis.item()

                    if id_ == 9: head_jt = True
                    if id_ == 8: neck_jt = True

                    if ((id_ == 6) or (id_ == 7)) and del_extra_jnts:
                        remove_pelvis_thorax_from_num_joints += 1

                    # Arrange ground truth in form {jnt: [[person1], [person2]]}
                    gt_per_joint = np.array([x, y, vis]).astype(np.float16)
                    gt_per_image[mpii_idx_to_jnt[id_]].append(gt_per_joint)

                    per_person_jnts.append(mpii_idx_to_jnt[id_])

                # If person 1 does not have rankl and person 2 has rankl, then prevent rankl being associated with p1
                # If jnt absent in person, then we append np.array([-1, -1, -1])
                all_jnts = set(list(mpii_idx_to_jnt.values()))
                per_person_jnts = set(per_person_jnts)
                jnt_absent_person = all_jnts - per_person_jnts
                for abs_joint in jnt_absent_person:
                    gt_per_image[abs_joint].append(np.array([-1, -1, -1]))

                num_joints_persons.append(num_joints - remove_pelvis_thorax_from_num_joints)
                scale.append(scale_img_mpii)
                objpos.append(objpose_img_mpii)

                # If both head and neck joint present, then move the head joint linearly towards the neck joint.
                if head_jt and neck_jt:
                    gt_per_image['head'][-1] = (lambda_head * gt_per_image['head'][-1])\
                                               + ((1 - lambda_head) * gt_per_image['neck'][-1])

                # Since annotation for atleast on person in image present, this flag will add GT to the dataset
                annotated_person_flag = True

            except KeyError:
                # Person 'x' could not have annotated joints, hence move to person 'y'
                continue

        if not annotated_person_flag:
            continue

        # Maintain compatibility with MPII and LSPET
        if del_extra_jnts:
            del gt_per_image['pelvis']
            del gt_per_image['thorax']

        # Add image, name, pred placeholder and gt
        img_dict['mpii']['img'].append(image)
        img_dict['mpii']['img_name'].append(img_name)
        img_dict['mpii']['img_pred'].append(mpii_template.copy())
        img_dict['mpii']['img_gt'].append(gt_per_image)
        img_dict['mpii']['normalizer'].append(normalizer_persons)
        img_dict['mpii']['dataset'].append('mpii')
        img_dict['mpii']['num_gt'].append(num_joints_persons)
        img_dict['mpii']['split'].append(train_test_mpii)
        img_dict['mpii']['scale'].append(scale)
        img_dict['mpii']['objpos'].append(objpos)
        img_dict['mpii']['num_ppl'].append(num_people)

    img_dict['mpii']['del_extra_jnts'] = del_extra_jnts
    img_dict['mpii']['lambda_head'] = lambda_head

    if mpii_only:
        string = 'mpii_only'
    else:
        string = 'lsp_lspet'

    np.save(file=os.path.join(root, 'cached', 'mpii_cache_{}.npy'.format(string)),
            arr=img_dict,
            allow_pickle=True)

    np.save(file=os.path.join(root, 'cached', 'mpii_maxPpl_cache_{}.npy'.format(string)),
            arr=max_person_in_img,
            allow_pickle=True)

    del img_dict['mpii']['del_extra_jnts']
    del img_dict['mpii']['lambda_head']

    return img_dict, max_person_in_img


def load_lspet(shuffle=False, train_ratio=0.7, conf=None):
    '''
    Similar to MPII, loads LSPET into a common dictionary format.
    :return: (dict) Dictionary containing the LSPET dataset
    '''

    lspet_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                        6: 'rwri', 7: 'relb', 8: 'rsho', 11: 'lwri', 10: 'lelb', 9: 'lsho',
                        12: 'neck', 13: 'head'}

    lspet_template = dict([(lspet_idx_to_jnt[i], []) for i in range(14)])

    img_dict = {'lspet': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'normalizer': [],
                          'dataset': [], 'num_gt': [], 'split': []}}

    annotation_lspet = scipy.io.loadmat(os.path.join(dataset_path[0], 'joints.mat'))['joints']  # Shape: 14,3,10000

    # 0: Train; 1: Validate
    # Load Train/Test split if conf.model_load_hg == True
    if conf.model_load_hg:
        train_test_split = np.load(os.path.join(conf.model_load_path, 'model_checkpoints/lspet_split.npy'))

    else:
        train_test_split = np.concatenate([np.zeros((int(annotation_lspet.shape[2] * train_ratio),), dtype=np.int8),
                                           np.ones((annotation_lspet.shape[2]
                                                    - int(annotation_lspet.shape[2] * train_ratio),), dtype=np.int8)],
                                          axis=0)

        if shuffle:
            logging.info('Shuffling LSPET')
            np.random.shuffle(train_test_split)

    np.save(file=conf.model_save_path.format('lspet_split.npy'), arr=train_test_split)

    # annotation_lspet.shape[2]
    for index in tqdm(range(annotation_lspet.shape[2])):
        image = plt.imread(os.path.join(dataset_path[0], 'images', filenames[0][index]))

        gt = annotation_lspet[:, :, index]
        gt_dict = dict([(lspet_idx_to_jnt[i], [gt[i]]) for i in range(gt.shape[0])])
        num_gt = sum([1 for i in range(gt.shape[0]) if gt[i][2]])

        # PCK@0.x : Normalizer
        lsho = gt[9, :2]
        rsho = gt[8, :2]
        lhip = gt[3, :2]
        rhip = gt[2, :2]
        torso_1 = np.linalg.norm(lsho - rhip)
        torso_2 = np.linalg.norm(rsho - lhip)
        torso = max(torso_1, torso_2)

        img_dict['lspet']['img'].append(image)
        img_dict['lspet']['img_name'].append(filenames[0][index])
        img_dict['lspet']['img_pred'].append(copy.deepcopy(lspet_template))
        img_dict['lspet']['img_gt'].append(gt_dict)
        img_dict['lspet']['normalizer'].append([torso])
        img_dict['lspet']['dataset'].append('lspet')
        img_dict['lspet']['num_gt'].append([num_gt])
        img_dict['lspet']['split'].append(train_test_split[index])

    return img_dict


def load_lsp(shuffle=False, train_ratio=0.7, conf=None):
    '''
    Similar to MPII, loads LSP into a common dictionary format.
    :return: (dict) Dictionary containing the LSP dataset
    '''

    lsp_idx_to_jnt = {0: 'rankl', 1: 'rknee', 2: 'rhip', 5: 'lankl', 4: 'lknee', 3: 'lhip',
                      6: 'rwri', 7: 'relb', 8: 'rsho', 11: 'lwri', 10: 'lelb', 9: 'lsho',
                      12: 'neck', 13: 'head'}

    lsp_template = dict([(lsp_idx_to_jnt[i], []) for i in range(14)])

    img_dict = {'lsp': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'normalizer': [],
                        'dataset': [], 'num_gt': [], 'split': []}}

    annotation_lsp = scipy.io.loadmat(os.path.join(dataset_path[2], 'joints.mat'))['joints']  # Shape: 3,14,2000

    # 0: Train; 1: Validate
    # Load Train/Test split if conf.model_load_hg == True
    if conf.model_load_hg:
        train_test_split = np.load(os.path.join(conf.model_load_path, 'model_checkpoints/lsp_split.npy'))

    else:
        train_test_split = np.concatenate([np.zeros((int(annotation_lsp.shape[2] * train_ratio),), dtype=np.int8),
                                           np.ones((annotation_lsp.shape[2]
                                                    - int(annotation_lsp.shape[2] * train_ratio),), dtype=np.int8)],
                                          axis=0)

        if shuffle:
            logging.info('Shuffling LSP')
            np.random.shuffle(train_test_split)

    np.save(file=conf.model_save_path.format('lsp_split.npy'), arr=train_test_split)

    # annotation_lsp.shape[2]
    for index in tqdm(range(annotation_lsp.shape[2])):
        image = plt.imread(os.path.join(dataset_path[2], 'images', filenames[2][index]))

        # Broadcasting rules apply: Toggle visibility of ground truth
        gt = abs(np.array([[0], [0], [1]]) - annotation_lsp[:, :, index])
        gt_dict = dict([(lsp_idx_to_jnt[i], [gt[:, i]]) for i in range(gt.shape[1])])
        num_gt = sum([1 for i in range(gt.shape[1]) if gt[:, i][2]])

        # PCK@0.x : Normalizer
        lsho = gt[:2, 9]
        rsho = gt[:2, 8]
        lhip = gt[:2, 3]
        rhip = gt[:2, 2]
        torso_1 = np.linalg.norm(lsho - rhip)
        torso_2 = np.linalg.norm(rsho - lhip)
        torso = max(torso_1, torso_2)


        img_dict['lsp']['img'].append(image)
        img_dict['lsp']['img_name'].append(filenames[2][index])
        img_dict['lsp']['img_pred'].append(copy.deepcopy(lsp_template))
        img_dict['lsp']['img_gt'].append(gt_dict)
        img_dict['lsp']['normalizer'].append([torso])
        img_dict['lsp']['dataset'].append('lsp')
        img_dict['lsp']['num_gt'].append([num_gt])
        img_dict['lsp']['split'].append(train_test_split[index])

    return img_dict


def load_hp_dataset(mpii=False, lspet=False, lsp=False, conf=None):
    '''
    Loads one of the three Human Pose datasets
    :param mpii: (bool) Load MPII
    :param lspet: (bool) Load LSPET
    :param lsp: (bool) Load LSP
    :param conf: (Object of ParseConfig)
    :return: (dict) Dataset
    '''

    assert mpii + lspet + lsp == 1, "One of MPII or LSPET or LSP needs to be selected for loading"

    global root, dataset_name, dataset_path, filenames

    root = Path(os.getcwd()).parent
    sys.path.append(root)

    dataset_name = ['lspet', 'mpii', 'lsp']
    dataset_path = list(map(lambda x: os.path.join(root, 'data', x), dataset_name))

    # [FORMAT] filenames: [[string, string, ... (lspet string)], [string, string, ... (mpii string)]]
    filenames_ = list(map(lambda path, name: open(os.path.join(path, '{}_filenames.txt'.format(name))),
                          dataset_path, dataset_name))
    filenames = list(map(lambda f: f.read().split(), filenames_))
    _ = list(map(lambda f: f.close(), filenames_))

    if mpii:
        logging.info('Loading MPII')
        mpii_params = conf.args['mpii_params']
        return load_mpii(lambda_head=mpii_params['lambda_head'], del_extra_jnts=mpii_params['del_extra_jnts'],
                         precached_mpii=conf.precached_mpii, mpii_only=conf.args['mpii_only'])

    elif lspet:
        logging.info('Loading LSPET')
        lspet_params = conf.args['lspet_params']
        return load_lspet(shuffle=lspet_params['shuffle'], train_ratio=lspet_params['train_ratio'], conf=conf)

    else:
        logging.info('Loading LSP')
        lsp_params = conf.args['lsp_params']
        return load_lsp(shuffle=lsp_params['shuffle'], train_ratio=lsp_params['train_ratio'], conf=conf)


class Dataset_MPII_LSPET_LSP(torch.utils.data.Dataset):

    def __init__(self, mpii_dict=None, lspet_dict=None, lsp_dict=None, activelearning_obj=None,
                 getitem_dump=None, conf=None, **kwargs):
        '''
        Implements the torch Dataset required by the torch dataLoader
        :param mpii_dict, lspet_dict, lsp_dict: (dict) Dataset dictionary from load_hp_dataset()
        :activelearning_obj (Object of class ActiveLearning)
        :param getitem_dump: (str) Model save path
        :param conf: (Object of ParseConfig)
        '''

        self.conf=conf
        self.viz = kwargs['misc']['viz']
        self.occlusion = kwargs['misc']['occlusion']
        self.hm_shape = kwargs['hourglass']['hm_shape']
        self.hm_peak = kwargs['misc']['hm_peak']
        self.threshold = kwargs['misc']['threshold'] * self.hm_peak
        self.model_save_path = getitem_dump

        # Training specific attributes:
        self.train_flag = False
        self.validate_flag = False
        self.model_input_dataset = None

        # Load dataset as class attributes
        self.mpii = mpii_dict['mpii']
        self.lspet = lspet_dict['lspet']
        self.lsp = lsp_dict['lsp']
        self.ind_to_jnt = list(ind_to_jnt.values())

        # Define active learning functions
        activelearning_samplers = {
            'random': activelearning_obj.random,
            'coreset': activelearning_obj.coreset_sampling,
            'learning_loss': activelearning_obj.learning_loss_sampling,
            'egl': activelearning_obj.expected_gradient_length_sampling,
            'entropy': activelearning_obj.multipeak_entropy
        }

        mpii_params = kwargs['mpii_params']

        # Create dataset by converting into numpy compatible types
        logging.info('Creating MPII dataset\n')
        self.mpii_dataset = self.create_mpii(train_ratio=mpii_params['train_ratio'], max_persons=mpii_params['max_persons'], shuffle=mpii_params['shuffle'])
        logging.info('Creating LSPET dataset\n')
        self.lspet_dataset = self.create_lspet(max_persons=mpii_params['max_persons'])
        logging.info('Creating LSP dataset\n')
        self.lsp_dataset = self.create_lsp(max_persons=mpii_params['max_persons'])

        # Train/validate splits
        logging.info('Splitting individual datasets into train and validation datasets\n')
        self.mpii_train, self.mpii_validate = self.create_train_validate(dataset=self.mpii_dataset)
        self.lspet_train, self.lspet_validate = self.create_train_validate(dataset=self.lspet_dataset)
        self.lsp_train, self.lsp_validate = self.create_train_validate(dataset=self.lsp_dataset)

        # Extract single person patches
        logging.info('Creating single person patches\n')
        self.mpii_train = self.mpii_single_person_extractor(train=True, max_persons=mpii_params['max_persons'])
        self.mpii_validate = self.mpii_single_person_extractor(validate=True, max_persons=mpii_params['max_persons'])
        logging.info('Size of Single person Train and Validate datasets: ')
        logging.info('Train: {}'.format(self.mpii_train['img'].shape[0]))
        logging.info('Validate: {}\n'.format(self.mpii_validate['img'].shape[0]))

        # Dataset sizes
        self.dataset_size = {'mpii': {'train': self.mpii_train['img'].shape[0], 'validation': self.mpii_validate['img'].shape[0]},
                             'lspet': {'train': self.lspet_train['img'].shape[0], 'validation': self.lspet_validate['img'].shape[0]},
                             'lsp': {'train': self.lsp_train['img'].shape[0], 'validation': self.lsp_validate['img'].shape[0]}}

        # Create train / validation data by combining individual components
        logging.info('Creating train and validation splits\n')

        if self.conf.args['mpii_only']:
            self.train_entire = self.merge_dataset(datasets=[self.mpii_train],
                                                   indices=[np.arange(self.mpii_train['img'].shape[0])]
                                                   )
            self.validate = self.merge_dataset(datasets=[self.mpii_validate],
                                               indices=[np.arange(self.mpii_validate['img'].shape[0])])

        else:
            self.train_entire = self.merge_dataset(datasets=[self.lspet_train, self.lsp_train],
                                                   indices=[np.arange(self.lspet_train['img'].shape[0]),
                                                            np.arange(self.lsp_train['img'].shape[0])]
                                                   )
            self.validate = self.merge_dataset(datasets=[self.lspet_validate, self.lsp_validate],
                                               indices=[np.arange(self.lspet_validate['img'].shape[0]),
                                                        np.arange(self.lsp_validate['img'].shape[0])])

        # Clearing RAM
        del self.mpii_train, self.mpii_validate, self.mpii_dataset, self.mpii,\
            self.lspet_train, self.lspet_validate, self.lspet_dataset, self.lspet,\
            self.lsp_train, self.lsp_validate, self.lsp_dataset, self.lsp,

        self.indices = activelearning_samplers[conf.active_learning_params['algorithm']](
            train=self.train_entire, dataset_size=self.dataset_size)

        self.train = self.merge_dataset(datasets=[self.train_entire], indices=[self.indices])

        logging.info('\nFinal size of Training Data: {}'.format(self.train['img'].shape[0]))
        logging.info('Final size of Validation Data: {}\n'.format(self.validate['img'].shape[0]))

        del self.train_entire

        # Decide which dataset is input to the model
        self.input_dataset(train=True)

        # Deciding augmentation techniques
        self.shift_scale_rotate = self.augmentation([albu.ShiftScaleRotate(p=1, shift_limit=0.15, scale_limit=0.25,
                                                                           rotate_limit=60, interpolation=cv2.INTER_LINEAR,
                                                                           border_mode=cv2.BORDER_CONSTANT, value=0)])

    def __len__(self):
        '''
        :return: Length of the dataset
        '''
        return self.model_input_dataset['gt'].shape[0]


    def __getitem__(self, i):
        '''
        :param i: Returns the i'th element in the dataset
        :return:
        '''

        image = self.model_input_dataset['img'][i]
        name = self.model_input_dataset['name'][i]
        gt = self.model_input_dataset['gt'][i]
        dataset = self.model_input_dataset['dataset'][i]
        num_gt = self.model_input_dataset['num_gt'][i]
        split = self.model_input_dataset['split'][i]
        num_persons = self.model_input_dataset['num_persons'][i]
        bbox_coords = self.model_input_dataset['bbox_coords'][i]
        normalizer = self.model_input_dataset['normalizer'][i]

        # Convert from XY cartesian to UV image coordinates
        xy_to_uv = lambda xy: (xy[1], xy[0])
        uv_to_xy = lambda uv: (uv[1], uv[0])

        # Determine crop
        img_shape = np.array(image.shape)
        [min_x, min_y, max_x, max_y] = bbox_coords[0]

        tl_uv = xy_to_uv(np.array([min_x, min_y]))
        br_uv = xy_to_uv(np.array([max_x, max_y]))
        min_u = tl_uv[0]
        min_v = tl_uv[1]
        max_u = br_uv[0]
        max_v = br_uv[1]

        centre = np.array([(min_u + max_u) / 2, (min_v + max_v) / 2])
        height = max_u - min_u
        width = max_v - min_v

        if self.train_flag:
            scale = np.random.uniform(low=1.5,high=2.5)
        else:
            scale = 2.0

        top_left = np.array([centre[0] - (scale * height / 2), centre[1] - (scale * width / 2)])
        bottom_right = np.array([centre[0] + (scale * height / 2), centre[1] + (scale * width / 2)])

        top_left = np.maximum(np.array([0, 0], dtype=np.int16), top_left.astype(np.int16))
        bottom_right = np.minimum(img_shape.astype(np.int16)[:-1], bottom_right.astype(np.int16))

        # Cropping the image and adjusting the ground truth
        image = image[top_left[0]: bottom_right[0], top_left[1]: bottom_right[1], :]
        for person in range(gt.shape[0]):
            for joint in range(gt.shape[1]):
                gt_uv = xy_to_uv(gt[person][joint])
                gt_uv = gt_uv - top_left
                gt[person][joint] = np.concatenate([gt_uv, np.array([gt[person][joint][2]])], axis=0)

        # Resize the image
        image, gt, scale_params = self.resize_image(image, gt, target_size=[256, 256, 3])
        gt[:, :, :2] = np.clip(a=gt[:, :, :2], a_min=0, a_max=255.9)

        # Augmentation
        if self.train_flag:
            augmented = self.shift_scale_rotate(image=image, keypoints=gt.reshape(-1, 3)[:, :2])
            image = augmented['image']
            gt[:, :, :2] = np.stack(augmented['keypoints'], axis=0).reshape(-1, self.conf.num_hm, 2)

        heatmaps, joint_exist = heatmap_generator(
            joints=np.copy(gt),occlusion=self.occlusion, hm_shape=self.hm_shape, img_shape=image.shape)

        heatmaps = self.hm_peak * heatmaps

        return torch.tensor(data=image/256.0, dtype=torch.float32, device='cpu'), \
               torch.tensor(data=heatmaps, dtype=torch.float32, device='cpu'),\
               gt, name, dataset, num_gt.astype(np.float32), split, num_persons,\
               scale_params, normalizer, joint_exist


    def create_mpii(self, train_ratio=0.7, max_persons=0, shuffle=False):
        '''
        Convert the dictionary dataset into a numpy indexing compatible dataset
        :param train_ratio: (Not needed if newell_validation in config.yml is True)
        :param max_persons: (int) Legacy reasons, ignore as the code is for single person only.
        :param shuffle: (bool)
        :return: (dictionary with values as numpy arrays)
        '''

        mpii = self.mpii

        assert len(mpii['img']) == len(mpii['img_name']) == len(mpii['img_gt']), \
            "MPII dataset image and labels mismatched."

        dataset = {'img': [], 'name': [], 'gt': -np.ones(shape=(len(mpii['img']), max_persons, self.conf.num_hm, 3)),
                   'dataset': [], 'num_gt': np.zeros(shape=(len(mpii['img']), max_persons)), 'split': [],
                   'num_persons': np.zeros(shape=(len(mpii['img']), 1)), 'normalizer': np.zeros(shape=(len(mpii['img']), max_persons)),
                   'bbox_coords': -np.ones(shape=(len(mpii['img']), max_persons, 4))}

        len_dataset = len(mpii['img'])

        for i in range(len_dataset):

            image = mpii['img'][i]
            image_name = mpii['img_name'][i]
            ground_truth = mpii['img_gt'][i]
            dataset_ = mpii['dataset'][i]
            num_gt = mpii['num_gt'][i]
            split = mpii['split'][i]
            normalizer = mpii['normalizer'][i]

            # Calculating the number of people
            num_ppl = 0
            for key in ground_truth.keys():
                num_ppl = max(num_ppl, len(ground_truth[key]))
                break   # All keys have same length, as if jnt absent in person, then we append np.array([-1, -1, -1])

            dataset['num_persons'][i] = num_ppl

            assert split == 1, "All annotated images should have split == 1"

            # Assigning to a Numpy Ground truth array
            for jnt in ground_truth.keys():
                for person_id in range(len(ground_truth[jnt])):
                    dataset['gt'][i, person_id, jnt_to_ind[jnt]] = ground_truth[jnt][person_id]

            # Assigning Bounding Box coordinates per person
            for person_id in range(num_ppl):

                x_coord = dataset['gt'][i, person_id, :, 0][np.where(dataset['gt'][i, person_id, :, 0] > -1)]
                y_coord = dataset['gt'][i, person_id, :, 1][np.where(dataset['gt'][i, person_id, :, 1] > -1)]

                min_x = np.min(x_coord)
                max_x = np.max(x_coord)
                min_y = np.min(y_coord)
                max_y = np.max(y_coord)

                dataset['bbox_coords'][i, person_id] = np.array([min_x, min_y, max_x, max_y])

            # Number of joints scaling factor
            for person_id in range(len(num_gt)):
                dataset['num_gt'][i, person_id] = num_gt[person_id]

            for person_id in range(len(normalizer)):
                dataset['normalizer'][i, person_id] = normalizer[person_id]

            dataset['img'].append(image)
            dataset['name'].append(image_name)
            dataset['dataset'].append(dataset_)
            dataset['split'].append(split)

        # Load Train/Test split if conf.model_load_hg = True
        if self.conf.model_load_hg:
            dataset['split'] = np.load(os.path.join(self.conf.model_load_path, 'model_checkpoints/mpii_split.npy'))

        else:
            # Create our own train/validation split for multi person dataset
            # 0: Train; 1: Validate

            if self.conf.mpii_newell_validation:
                logging.info('\nCreating the Newell validation split.\n')
                with open(os.path.join(root, 'cached', 'Stacked_HG_ValidationImageNames.txt')) as valNames:
                    valNames_ = [x.strip('\n') for x in valNames.readlines()]

                assert len_dataset == len(dataset['name']), "Mismatch in number of images and image names."

                dataset['split'] = np.array([1 if x in valNames_ else 0 for x in dataset['name']])
                # assert np.sum(dataset['split']) == len(valNames_)
                # "THIS ASSERTION WILL NOT HOLD TRUE. Newell list has duplicates."

            else:
                dataset['split'] = np.concatenate([np.zeros(int(len_dataset*train_ratio),),
                                                   np.ones((len_dataset - int(len_dataset*train_ratio)),)],
                                                  axis=0)
                if shuffle: np.random.shuffle(dataset['split'])

        np.save(file=self.model_save_path.format('mpii_split.npy'), arr=dataset['split'])

        dataset['img'] = np.array(dataset['img'])
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])

        logging.info('MPII dataset description:')
        logging.info('Length (#images): {}\t(#gt) {}'.format(len(dataset['img']), dataset['gt'].shape[0]))

        return dataset


    def create_lspet(self, max_persons=1):
        '''
        Convert the dictionary dataset into a numpy indexing compatible dataset
        :return: (dictionary with values as numpy arrays)
        '''
        lspet = copy.deepcopy(self.lspet)
        assert len(lspet['img']) == len(lspet['img_name']) == len(lspet['img_gt']), \
            "LSPET dataset image and labels mismatched."

        dataset = {'img': [], 'name': [], 'gt': -np.ones(shape=(len(lspet['img']), max_persons, 14, 3)),
                   'dataset': [], 'num_gt': np.zeros(shape=(len(lspet['img']), max_persons)), 'split': [],
                   'num_persons': np.ones(shape=(len(lspet['img']), 1)), 'normalizer': np.zeros(shape=(len(lspet['img']), max_persons)),
                   'bbox_coords': -np.ones(shape=(len(lspet['img']), max_persons, 4))}    # max_persons is always 1 for lsp*

        len_dataset = len(lspet['img'])
        for i in range(len_dataset):

            image = lspet['img'][i]
            image_name = lspet['img_name'][i]
            ground_truth = lspet['img_gt'][i]
            dataset_ = lspet['dataset'][i]
            num_gt = lspet['num_gt'][i]
            split = lspet['split'][i]
            normalizer = lspet['normalizer'][i]

            for jnt in ground_truth.keys():
                dataset['gt'][i, 0, jnt_to_ind[jnt]] = ground_truth[jnt][0]

            # Assigning Bounding Box coordinates per person
            x_coord = dataset['gt'][i, 0, :, 0][np.where(dataset['gt'][i, 0, :, 2] == 1)]
            y_coord = dataset['gt'][i, 0, :, 1][np.where(dataset['gt'][i, 0, :, 2] == 1)]

            x_coord = x_coord[np.where(x_coord > -1)]
            y_coord = y_coord[np.where(y_coord > -1)]

            min_x = np.min(x_coord)
            max_x = np.max(x_coord)
            min_y = np.min(y_coord)
            max_y = np.max(y_coord)

            dataset['bbox_coords'][i, 0] = np.array([min_x, min_y, max_x, max_y])

            # Assigning number of GT to person 0
            dataset['num_gt'][i, 0] = num_gt[0]

            dataset['normalizer'][i, 0] = normalizer[0]

            dataset['img'].append(image)
            dataset['name'].append(image_name)
            dataset['dataset'].append(dataset_)
            dataset['split'].append(split)

        dataset['img'] = np.array(dataset['img'])
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])
        dataset['split'] = np.array(dataset['split'])

        logging.info('LSPET dataset description:')
        logging.info('Length (#images): {}\t(#gt) {}'.format(len(dataset['img']), dataset['gt'].shape[0]))

        return dataset


    def create_lsp(self, max_persons=1):
        '''
        Convert the dictionary dataset into a numpy indexing compatible dataset
        :return: (dictionary with values as numpy arrays)
        '''
        lsp = copy.deepcopy(self.lsp)
        assert len(lsp['img']) == len(lsp['img_name']) == len(lsp['img_gt']), \
            "LSP dataset image and labels mismatched."

        dataset = {'img': [], 'name': [], 'gt': -np.ones(shape=(len(lsp['img']), max_persons, 14, 3)),
                   'dataset': [], 'num_gt': np.zeros(shape=(len(lsp['img']), max_persons)), 'split': [],
                   'num_persons': np.ones(shape=(len(lsp['img']), 1)), 'normalizer': np.zeros(shape=(len(lsp['img']), max_persons)),
                   'bbox_coords': -np.ones(shape=(len(lsp['img']), max_persons, 4))}  # max_persons is always 1 for lsp*

        len_dataset = len(lsp['img'])
        for i in range(len_dataset):

            image = lsp['img'][i]
            image_name = lsp['img_name'][i]
            ground_truth = lsp['img_gt'][i]
            dataset_ = lsp['dataset'][i]
            num_gt = lsp['num_gt'][i]
            split = lsp['split'][i]
            normalizer = lsp['normalizer'][i]

            for jnt in ground_truth.keys():
                dataset['gt'][i, 0, jnt_to_ind[jnt]] = ground_truth[jnt][0]

            # Assigning Bounding Box coordinates per person
            x_coord = dataset['gt'][i, 0, :, 0][np.where(dataset['gt'][i, 0, :, 0] > -1)]
            y_coord = dataset['gt'][i, 0, :, 1][np.where(dataset['gt'][i, 0, :, 1] > -1)]

            min_x = np.min(x_coord)
            max_x = np.max(x_coord)
            min_y = np.min(y_coord)
            max_y = np.max(y_coord)

            dataset['bbox_coords'][i, 0] = np.array([min_x, min_y, max_x, max_y])

            dataset['num_gt'][i, 0] = num_gt[0]

            dataset['normalizer'][i, 0] = normalizer[0]

            dataset['img'].append(image)
            dataset['name'].append(image_name)
            dataset['dataset'].append(dataset_)
            dataset['split'].append(split)

        dataset['img'] = np.array(dataset['img'])
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])
        dataset['split'] = np.array(dataset['split'])

        logging.info('LSP dataset description:')
        logging.info('Length (#images): {}\t(#gt) {}'.format(len(dataset['img']), dataset['gt'].shape[0]))

        return dataset


    def create_train_validate(self, dataset=None):
        '''
        Separate common dataset into train/validate based on split
        :param (dict) Dictionary containing numpy arrays from one of the create_* methods
        :return: (dict, dict) Train and Validation datasets
        '''

        # Separate train and validate
        train_idx = []
        val_idx = []
        for i in range(len(dataset['img'])):
            if dataset['split'][i] == 0:
                train_idx.append(i)
            else:
                assert dataset['split'][i] == 1, \
                    "Split has value: {}, should be either 0 or 1".format(dataset['split'][i])
                val_idx.append(i)

        train_dataset = {}
        val_dataset = {}
        for key in dataset.keys():
            train_dataset[key] = dataset[key][train_idx]
            val_dataset[key] = dataset[key][val_idx]

        return train_dataset, val_dataset


    def merge_dataset(self, datasets=None, indices=None):
        '''
        Combines datasets
        :param datasets: (list) List containing dataset dictionaries
        :param indices: (list to support index slicing of individual datasets)
        :return:
        '''
        assert type(datasets) == list and len(datasets) != 0
        assert len(datasets) == len(indices)

        for i in range(len(datasets) - 1):
            assert datasets[i].keys() == datasets[i+1].keys(), "Dataset keys do not match"

        # Merge datasets
        merged_dataset = {}
        for key in datasets[0].keys():
            merged_dataset[key] = np.concatenate([data[key][index_] for index_, data in zip(indices, datasets)], axis=0)

        # Sampling based on indices
        merged_dataset['index'] = np.arange(merged_dataset['img'].shape[0])

        return merged_dataset


    def recreate_images(self, gt=False, pred=False, train=False, validate=False, external=False, ext_data=None):
        '''
        Method to support visualizing images with pred, gt on the image.
        Returns a dictionary that can then be given to visualize_image() in utils
        Refer to visualize_predictions() in main.py
        :return: (dict) Compatible with visualize_images in utils
        '''
        assert gt + pred != 0, "Specify atleast one of GT or Pred"
        assert train + validate + external == 1,\
            "Can create visualize_image compatible arrays only for train/val in one function call."

        if external:
            assert ext_data, "ext_dataset can't be none to recreate external datasets"
            data_split = ext_data
        elif train:
            data_split = self.train
        else:
            data_split = self.validate

        # Along with the below entries, we also pass bbox coordinates for each dataset
        img_dict = {'mpii': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'split': [], 'dataset': []},
                    'lspet': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'split': [], 'dataset': []},
                    'lsp': {'img': [], 'img_name': [], 'img_pred': [], 'img_gt': [], 'split': [], 'dataset': []}}

        for i in range(len(data_split['img'])):
            dataset = data_split['dataset'][i]
            img_dict[dataset]['img'].append(data_split['img'][i])
            img_dict[dataset]['img_name'].append(data_split['name'][i])
            img_dict[dataset]['split'].append(data_split['split'][i])
            img_dict[dataset]['dataset'].append(data_split['dataset'][i])

            try:
                img_dict[dataset]['bbox_coords'] = np.concatenate([img_dict[dataset]['bbox_coords'],
                                                                   data_split['bbox_coords'][i].reshape(1, -1, 4)], axis=0)
            except KeyError:
                img_dict[dataset]['bbox_coords'] = data_split['bbox_coords'][i].reshape(1, -1, 4)

            joint_dict = dict([(ind_to_jnt[i], []) for i in range(self.conf.num_hm)])
            gt_dict = copy.deepcopy(joint_dict)
            pred_dict = copy.deepcopy(joint_dict)

            if gt:
                for person in range(int(data_split['num_persons'][i, 0])):
                    for joint in range(self.conf.num_hm):
                        gt_dict[ind_to_jnt[joint]].append(data_split['gt'][i, person, joint])

            if pred:
                for person in range(int(data_split['num_persons'][i, 0])):
                    for joint in range(self.conf.num_hm):
                        pred_dict[ind_to_jnt[joint]].append(data_split['pred'][i, person, joint])

            img_dict[dataset]['img_gt'].append(gt_dict)
            img_dict[dataset]['img_pred'].append(pred_dict)

        return img_dict


    def input_dataset(self, train=False, validate=False):
        '''
        Switch between training and validation split on-the-fly
        :param train: (bool) Select train
        :param validate: (bool) Select validation dataset
        :return: NoneType
        '''
        assert train + validate == 1, "Either one of train or validate_entire needs to be chosen"

        if train:
            self.model_input_dataset = self.train
            self.train_flag = True
            self.validate_flag = False
        else:
            self.model_input_dataset = self.validate
            self.validate_flag = True
            self.train_flag = False

        return None


    def mpii_single_person_extractor(self, train=False, validate=False, max_persons=None):
        '''
        Extract single persons in the MPII dataset
        :param train: (bool) Extract single persons for train
        :param validate: (bool) Extract single persons for validation split
        :param max_persons: (int)
        :return:
        '''
        assert train + validate == 1, "Only one of Train or Validate can be true at any given time"
        if train:
            mpii_dataset = self.mpii_train
        else:
            mpii_dataset = self.mpii_validate

        dataset = {'img': [], 'name': [], 'gt': np.empty(shape=(0, max_persons, self.conf.num_hm, 3)),
                   'dataset': [], 'num_gt': np.empty(shape=(0, max_persons)), 'split': [],
                   'num_persons': np.empty(shape=(0, 1)), 'normalizer': np.empty(shape=(0, max_persons)),
                   'bbox_coords': np.empty(shape=(0, max_persons, 4))}

        for i in range(len(mpii_dataset['img'])):
            for p in range(int(mpii_dataset['num_persons'][i][0])):
                dataset['img'].append(mpii_dataset['img'][i])
                dataset['name'].append(mpii_dataset['name'][i][:-4] + '_{}.jpg'.format(p))
                dataset['dataset'].append(mpii_dataset['dataset'][i])
                dataset['split'].append(mpii_dataset['split'][i])

                gt_ = np.zeros_like(mpii_dataset['gt'][i])
                gt_[0] = mpii_dataset['gt'][i, p]
                dataset['gt'] = np.concatenate([dataset['gt'], gt_.reshape(1, max_persons, self.conf.num_hm, 3)],
                                               axis=0)

                num_gt_ = np.zeros_like(mpii_dataset['num_gt'][i])
                num_gt_[0] = mpii_dataset['num_gt'][i, p]
                dataset['num_gt'] = np.concatenate([dataset['num_gt'], num_gt_.reshape(1, max_persons)],
                                               axis=0)

                normalizer_ = np.zeros_like(mpii_dataset['normalizer'][i])
                normalizer_[0] = mpii_dataset['normalizer'][i, p]
                dataset['normalizer'] = np.concatenate([dataset['normalizer'], normalizer_.reshape(1, max_persons)],
                                                       axis=0)

                dataset['num_persons'] = np.concatenate([dataset['num_persons'], np.array([1]).reshape(1, 1)],
                                               axis=0)

                bbox_ = np.zeros_like(mpii_dataset['bbox_coords'][i])
                bbox_[0] = mpii_dataset['bbox_coords'][i, p]
                dataset['bbox_coords'] = np.concatenate([dataset['bbox_coords'],bbox_.reshape(1, max_persons, 4)],
                                               axis=0)

        dataset['img'] = np.array(dataset['img'])
        dataset['name'] = np.array(dataset['name'])
        dataset['dataset'] = np.array(dataset['dataset'])
        dataset['split'] = np.array(dataset['split'])

        return dataset


    def resize_image(self, image_=None, gt=None, target_size=None):
        '''
        Helper method to scale the images
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

        gt *= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)
        gt[:, :, 0] += padding_u
        gt[:, :, 1] += padding_v

        scale_params = {'scale_factor': scale_factor, 'padding_u': padding_u, 'padding_v': padding_v}

        return output_img, gt, scale_params


    def augmentation(self, transform):
        '''
        Helper method that defines the augmentation
        :param transformation:
        :return:
        '''
        return albu.Compose(transform, p=1, keypoint_params=albu.KeypointParams(format='yx', remove_invisible=False))


    def estimate_uv(self, hm_array, pred_placeholder):
        '''
        Estimate the uv location
        :param hm_array: (numpy ndarray) Infer UV location from heatmap
        :param pred_placeholder: (numpy ndarray) Stores the uv locations
        :return: (numpy ndarray) same as pred_placeholder
        '''
        # Iterate over each heatmap
        for jnt_id in range(hm_array.shape[0]):
            pred_placeholder[0, jnt_id, :] = uv_from_heatmap(hm=hm_array[jnt_id], threshold=self.threshold)
        return pred_placeholder


    def upscale(self, joints, scale_params):
        '''
        Helper method to upscale the joints
        :return:
        '''
        joints[:, :, 0] -= scale_params['padding_u']
        joints[:, :, 1] -= scale_params['padding_v']
        joints /= np.array([scale_params['scale_factor'], scale_params['scale_factor'], 1]).reshape(1, 1, 3)

        return joints