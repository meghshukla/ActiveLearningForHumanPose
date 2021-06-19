import os
import copy
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau as ReduceLROnPlateau

from config import ParseConfig
from utils import visualize_image
from utils import heatmap_loss
from activelearning import ActiveLearning
from dataloader import load_hp_dataset
from dataloader import Dataset_MPII_LSPET_LSP
from evaluation import PercentageCorrectKeypoint
from models.learning_loss.LearningLoss import LearnLossActive
from models.stacked_hourglass.StackedHourglass import PoseNet as Hourglass

# Global declarations
logging.getLogger().setLevel(logging.INFO)
os.chdir(os.path.dirname(os.path.realpath(__file__)))


class Train(object):
    def __init__(self, network, learnloss, hyperparameters, dataset_obj, conf, tb_writer):
        '''
        Class for training the model
        Training will train the Hourglass module

        :param network: (torch.nn) Hourglass model
        :param llal_ntwk: (torch.nn) Learning Loss model
        :param hyperparameters: (dict) Various hyperparameters used in training
        :param loc_object: (Object of LocalizationLoader) Controls the data fed into torch_dataloader
        :param model_save_path (string) The path directory where the training output will be logged.
        :param conf: (Object of ParseConfig) Contains the configurations for the model
        :param tb_writer: (Object of SummaryWriter) Tensorboard writer to log values
        :param wt_reg: (Bool) Whether to use weight regularization or not
        '''

        # Dataset Settings
        self.dataset_obj = dataset_obj
        self.tb_writer = tb_writer                                           # Tensorboard writer
        self.network = network                                               # Hourglass network
        self.batch_size = conf.batch_size
        self.epoch = hyperparameters['num_epochs']
        self.hyperparameters = hyperparameters
        self.model_save_path = conf.model_save_path
        self.optimizer = hyperparameters['optimizer']                        # Adam / SGD
        self.loss_fn = hyperparameters['loss_fn']                            # MSE
        self.learning_rate = hyperparameters['optimizer_config']['lr']
        self.start_epoch = hyperparameters['start_epoch']                    # Used in case of resume training
        self.num_hm = conf.num_hm                                            # Number of heatmaps
        self.joint_names = self.dataset_obj.ind_to_jnt
        self.hg_depth = 4                                                    # Depth of hourglass
        self.n_stack = conf.n_stack

        self.train_learning_loss = conf.train_learning_loss
        self.learnloss_network = learnloss
        self.learnloss_margin = conf.learning_loss_margin
        self.learnloss_warmup = conf.learning_loss_warmup
        self.learnloss_original = conf.learning_loss_original
        self.learnloss_obj = conf.learning_loss_obj

        # Stacked Hourglass scheduling
        if self.train_learning_loss:
            min_lr = [0.000003, conf.lr]
        else:
            min_lr = 0.000003

        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.5, patience=8, cooldown=2, min_lr=min_lr, verbose=True)

        self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, self.batch_size,
                                                            shuffle=True, num_workers=8, drop_last=True)

        if torch.cuda.device_count() > 1:
            cuda_devices = [torch.device('cuda:0'), torch.device('cuda:1')]
        else:
            cuda_devices = [torch.device('cuda:0'), torch.device('cuda:0')]

        self.cuda_devices = cuda_devices
        if conf.learnloss_only:
            self.train_hg_bool = torch.tensor(0.0).cuda(cuda_devices[-1])
        else:
            self.train_hg_bool = torch.tensor(1.0).cuda(cuda_devices[-1])


    def train_model(self):
        '''
        Training Loop: Hourglass and/or Learning Loss
        :return: None
        '''

        print("Initializing training: Epochs - {}\tBatch Size - {}".format(self.hyperparameters['num_epochs'],
                                                                           self.batch_size))

        best_val_hg = np.inf
        best_val_learnloss = np.inf
        best_epoch_hg = -1
        best_epoch_learnloss = -1
        global_step = 0

        # Variable to store all the loss values for logging
        loss_across_epochs = []
        validation_across_epochs = []

        for e in range(self.start_epoch, self.epoch):
            epoch_loss = []
            epoch_loss_learnloss = []

            # Network alternates between train() and validate()
            self.network.train()
            if self.train_learning_loss:
                self.learnloss_network.train()

            self.dataset_obj.input_dataset(train=True)

            # Training loop
            logging.info('Training for epoch: {}'.format(e+1))
            for (images, heatmaps, _, _, _, gt_per_image, split, _, _, _, joint_exist) in tqdm(self.torch_dataloader):

                assert split[0] == 0, "Training split should be 0."

                gt_per_image = gt_per_image.to(non_blocking=True, device=self.cuda_devices[-1])

                # Will clear the gradients of hourglass
                self.optimizer.zero_grad()
                outputs, hourglass_features = self.network(images)

                heatmaps = heatmaps.to(non_blocking=True, device=self.cuda_devices[-1])
                loss = heatmap_loss(outputs, heatmaps, self.n_stack)

                learning_loss_ = loss.clone().detach()
                learning_loss_ = torch.mean(learning_loss_, dim=[1])

                loss = (torch.mean(loss)) * self.train_hg_bool
                self.tb_writer.add_scalar('Train/Loss_batch', torch.mean(loss), global_step)

                loss.backward()

                # Train the learning loss network
                if self.train_learning_loss:
                    loss_learnloss = self.learning_loss(hourglass_features, learning_loss_, self.learnloss_margin, gt_per_image, e)
                    loss_learnloss.backward()
                    epoch_loss_learnloss.append(loss_learnloss.cpu().data.numpy())

                # Weight update
                self.optimizer.step()
                global_step += 1

                # Store the loss per batch
                epoch_loss.append(loss.cpu().data.numpy())

            epoch_loss = np.mean(epoch_loss)
            if self.train_learning_loss:
                epoch_loss_learnloss = np.mean(epoch_loss_learnloss)

            # Returns average validation loss per element
            if self.train_learning_loss:
                validation_loss_hg, validation_learning_loss = self.validation(e)
            else:
                validation_loss_hg = self.validation(e)
                validation_learning_loss = 0.0

            # Learning rate scheduler on the HourGlass validation loss
            self.scheduler.step(validation_loss_hg)

            # TensorBoard Summaries
            self.tb_writer.add_scalar('Train', torch.Tensor([epoch_loss]), global_step)
            self.tb_writer.add_scalar('Validation/HG_Loss', torch.Tensor([validation_loss_hg]), global_step)
            if self.train_learning_loss:
                self.tb_writer.add_scalar('Validation/Learning_Loss', torch.Tensor([validation_learning_loss]), global_step)

            # Save the model
            torch.save(self.network.state_dict(),
                       self.model_save_path.format("model_epoch_{}.pth".format(e + 1)))

            if self.train_learning_loss:
                torch.save(self.learnloss_network.state_dict(),
                           self.model_save_path.format("model_epoch_{}_learnloss.pth".format(e + 1)))

            # For resume training ONLY:
            # If learn_loss, then optimizer will have two param groups
            # Hence during load, ensure llal module is loaded/not loaded
            torch.save({'epoch': e + 1,
                        'optimizer_load_state_dict': self.optimizer.state_dict(),
                        'mean_loss': epoch_loss,
                        'mean_loss_validation': {'HG': validation_loss_hg, 'LearningLoss': validation_learning_loss},
                        'learn_loss': self.train_learning_loss},
                        self.model_save_path.format("optim_epoch_{}.tar".format(e + 1)))

            # Save if best model
            if best_val_hg > validation_loss_hg:
                torch.save(self.network.state_dict(),
                           self.model_save_path.format("best_model.pth"))

                torch.save(self.learnloss_network.state_dict(),
                           self.model_save_path.format("best_model_learnloss_hg.pth"))

                best_val_hg = validation_loss_hg
                best_epoch_hg = e + 1

                torch.save({'epoch': e + 1,
                            'optimizer_load_state_dict': self.optimizer.state_dict(),
                            'mean_loss_train': epoch_loss,
                            'mean_loss_validation': {'HG': validation_loss_hg, 'LearningLoss': validation_learning_loss},
                            'learn_loss': self.train_learning_loss},
                           self.model_save_path.format("optim_best_model.tar"))

            if self.train_learning_loss:
                if best_val_learnloss > validation_learning_loss and validation_learning_loss != 0.0:
                    torch.save(self.learnloss_network.state_dict(),
                               self.model_save_path.format("best_model_learnloss_{}.pth".format(self.learnloss_obj)))

                    best_val_learnloss = validation_learning_loss
                    best_epoch_learnloss = e + 1

            print("Loss at epoch {}/{}: (train) {}\t"
                  "Learning Loss: (train) {}\t"
                  "(validation: HG) {}\t"
                  "(Validation: Learning Loss) {}\t"
                  "(Best Model) {}".format(
                e+1,
                self.epoch,
                epoch_loss,
                epoch_loss_learnloss,
                validation_loss_hg,
                validation_learning_loss,
                best_epoch_hg))

            loss_across_epochs.append(epoch_loss)
            validation_across_epochs.append(validation_loss_hg)

            # Save the loss values
            f = open(self.model_save_path.format("loss_data.txt"), "w")
            f_ = open(self.model_save_path.format("validation_data.txt"), "w")
            f.write("\n".join([str(lsx) for lsx in loss_across_epochs]))
            f_.write("\n".join([str(lsx) for lsx in validation_across_epochs]))
            f.close()
            f_.close()

        self.tb_writer.close()
        logging.info("Model training completed\nBest validation loss (HG): {}\tBest Epoch: {}"
                     "\nBest validation loss (LLAL): {}\tBest Epoch: {}".format(
            best_val_hg, best_epoch_hg, best_val_learnloss, best_epoch_learnloss))


    def validation(self, e):
        '''
        Validation loss
        :param e: (int) Epoch
        :return: (Float): Mean validation loss per batch for Hourglass and Learning Loss (if LL activated in inc_config file.)
        '''
        with torch.no_grad():
            # Stores the loss for all batches
            epoch_val_hg = []

            if self.train_learning_loss:
                epoch_val_learnloss = []

            self.network.eval()
            if self.train_learning_loss:
                self.learnloss_network.eval()

            # Augmentation only needed in Training
            self.dataset_obj.input_dataset(validate=True)

            # Compute and store batch-wise validation loss in a list
            logging.info('Validation for epoch: {}'.format(e+1))
            for (images, heatmaps, _, _, _, gt_per_img, split, _, _, _, joint_exist) in tqdm(self.torch_dataloader):

                assert split[0] == 1, "Validation split should be 1."

                gt_per_img = gt_per_img.to(non_blocking=True, device=self.cuda_devices[-1])

                outputs, hourglass_features = self.network(images)

                heatmaps = heatmaps.to(non_blocking=True, device=self.cuda_devices[-1])

                loss_val_hg = heatmap_loss(outputs, heatmaps, self.n_stack)

                learning_loss_val = loss_val_hg.clone().detach()
                learning_loss_val = torch.mean(learning_loss_val, dim=[1])

                loss_val_hg = torch.mean(loss_val_hg)
                epoch_val_hg.append(loss_val_hg.cpu().data.numpy())

                if self.train_learning_loss:
                    loss_val_learnloss = self.learning_loss(hourglass_features, learning_loss_val, self.learnloss_margin, gt_per_img, e)
                    epoch_val_learnloss.append(loss_val_learnloss.cpu().data.numpy())

            print("Validation Loss HG at epoch {}/{}: {}".format(e+1, self.epoch, np.mean(epoch_val_hg)))

            if self.train_learning_loss:
                print("Validation Learning Loss at epoch {}/{}: {}".format(e+1, self.epoch, np.mean(epoch_val_learnloss)))
                return np.mean(epoch_val_hg), np.mean(epoch_val_learnloss)

            else:
                return np.mean(epoch_val_hg)


    def learning_loss(self, hg_encodings, true_loss, margin, gt_per_img, epoch):
        '''
        Learning Loss module
        Refer:
        1. "Learning Loss For Active Learning, CVPR 2019"
        2. "A Mathematical Analysis of Learning Loss for Active Learning in Regression, CVPRW 2021"

        :param hg_encodings: (Dict of tensors) Intermediate (Hourglass) and penultimate layer output of the Hourglass network
        :param true_loss: (Tensor of shape [Batch Size]) Loss computed from HG prediction and ground truth
        :param margin: (scalar) tolerance margin between predicted losses
        :param gt_per_img: (Tensor, shape [Batch Size]) Number of ground truth per image
        :param epoch: (scalar) Epoch, used in learning loss warm start-up
        :return: (Torch scalar tensor) Learning Loss
        '''

        # Concatenate the layers instead of a linear combination
        with torch.no_grad():
            if self.learnloss_original:
                # hg_depth == 4 means depth is {1, 2, 3, 4}. If we want depth 5, range --> (1, 4+2)
                # encodings = torch.cat([hg_encodings[depth] for depth in range(1, self.hg_depth + 2)], dim=-1)
                encodings = hg_encodings['penultimate']

            else:
                # No longer concatenating, will now combine features through convolutional layers
                encodings = torch.cat([hg_encodings['feature_5'].reshape(self.batch_size, hg_encodings['feature_5'].shape[1], -1),               # 64 x 64
                                       hg_encodings['feature_4'].reshape(self.batch_size, hg_encodings['feature_4'].shape[1], -1),               # 32 x 32
                                       hg_encodings['feature_3'].reshape(self.batch_size, hg_encodings['feature_3'].shape[1], -1),               # 16 x 16
                                       hg_encodings['feature_2'].reshape(self.batch_size, hg_encodings['feature_2'].shape[1], -1),               # 8 x 8
                                       hg_encodings['feature_1'].reshape(self.batch_size, hg_encodings['feature_1'].shape[1], -1)], dim=2)       # 4 x 4

        emperical_loss, encodings = self.learnloss_network(encodings)
        emperical_loss = emperical_loss.squeeze()

        assert emperical_loss.shape == true_loss.shape, "Mismatch in Batch size for true and emperical loss"

        with torch.no_grad():
            # Scale the images as per the number of joints
            # To prevent DivideByZero. PyTorch does not throw an exception to DivideByZero
            gt_per_img = torch.sum(gt_per_img, dim=1)
            gt_per_img += 0.1
            if self.learnloss_obj == 'prob':
                true_loss = true_loss / gt_per_img

            # Splitting into pairs: (i, i+half)
            half_split = true_loss.shape[0] // 2

            true_loss_i = true_loss[: half_split]
            true_loss_j = true_loss[half_split: 2 * half_split]

        emp_loss_i = emperical_loss[: (emperical_loss.shape[0] // 2)]
        emp_loss_j = emperical_loss[(emperical_loss.shape[0] // 2): 2 * (emperical_loss.shape[0] // 2)]

        # Loss according to CVPR '19
        if self.learnloss_obj == 'pair':
            loss_sign = torch.sign(true_loss_i - true_loss_j)
            loss_emp = (emp_loss_i - emp_loss_j)

            # Learning Loss objective
            llal_loss = torch.max(torch.zeros(half_split, device=loss_sign.device), (-1 * (loss_sign * loss_emp)) + margin)

        # Loss according to CVPR '21
        elif self.learnloss_obj == 'prob':
            with torch.no_grad():
                true_loss_ = torch.cat([true_loss_i.reshape(-1, 1), true_loss_j.reshape(-1, 1)], dim=1)
                true_loss_scaled = true_loss_ / torch.sum(true_loss_, dim=1, keepdim=True)

            emp_loss_ = torch.cat([emp_loss_i.reshape(-1, 1), emp_loss_j.reshape(-1, 1)], dim=1)
            emp_loss_logsftmx = torch.nn.LogSoftmax(dim=1)(emp_loss_)
            llal_loss = torch.nn.KLDivLoss(reduction='batchmean')(input=emp_loss_logsftmx, target=true_loss_scaled)

        else:
            raise NotImplementedError('Currently only "pair" or "prob" supported. ')

        if self.learnloss_warmup <= epoch:
            return torch.mean(llal_loss)
        else:
            return 0.0 * torch.mean(llal_loss)


class Metric(object):
    def __init__(self, network, dataset_obj, conf):
        '''
        Class for Testing the model:
            1. Compute ground truth and predictions
            2. Computing metrics: PCK@0.x
        :param network: (torch.nn) Hourglass network to compute predictions
        :param dataset_obj: (Dataset object) Handles data to be fed to PyTorch DataLoader
        :param conf: (Object of ParseConfig) Configuration for the experiment
        '''

        self.dataset_obj = dataset_obj
        self.dataset_obj.input_dataset(validate=True)

        self.network = network
        self.model_save_path = conf.model_save_path
        self.viz=conf.args['misc']['viz']                         # Controls visualization
        self.conf = conf
        self.epoch = conf.model_load_epoch

        self.ind_to_jnt = {0: 'head', 1: 'neck', 2: 'lsho', 3: 'lelb', 4: 'lwri', 5: 'rsho', 6: 'relb', 7: 'rwri',
                           8: 'lhip', 9: 'lknee', 10: 'lankl', 11: 'rhip', 12: 'rknee', 13: 'rankl'}

        if conf.num_hm == 16:
            self.ind_to_jnt[14] = 'pelvis'
            self.ind_to_jnt[15] = 'thorax'

        self.torch_dataloader = torch.utils.data.DataLoader(self.dataset_obj, conf.batch_size, shuffle=False,
                                                            num_workers=8)

    def inference(self):
        '''
        Returns model inferences
        :return: (dict) images, heatmaps and other information to obtain U, V
        '''

        self.network.eval()
        logging.info("Starting model inference")

        image_ = None
        outputs_ = None
        scale_ = None
        num_gt_ = None
        dataset_ = None
        name_ = None
        gt_ = None
        normalizer_ = None

        with torch.no_grad():
            for (images, _, gt, name, dataset, num_gt, split, _, scale_params, normalizer, joint_exist) in tqdm(
                    self.torch_dataloader):

                assert split[0] == 1, "Validation split should be 1."

                outputs, hourglass_features = self.network(images)

                outputs = outputs[:, -1].detach()

                try:
                    image_ = torch.cat((image_, images.clone()), dim=0)
                    outputs_ = torch.cat((outputs_, outputs.cpu().clone()), dim=0)
                    scale_['scale_factor'] = torch.cat((scale_['scale_factor'], scale_params['scale_factor']), dim=0)
                    scale_['padding_u'] = torch.cat((scale_['padding_u'], scale_params['padding_u']), dim=0)
                    scale_['padding_v'] = torch.cat((scale_['padding_v'], scale_params['padding_v']), dim=0)
                    num_gt_ = torch.cat((num_gt_, num_gt), dim=0)
                    dataset_ = dataset_ + dataset
                    name_ = name_ + name
                    gt_ = torch.cat((gt_, gt), dim=0)
                    normalizer_ = torch.cat((normalizer_, normalizer), dim=0)

                except TypeError:
                    image_ = images.clone()
                    outputs_ = outputs.cpu().clone()
                    scale_ = copy.deepcopy(scale_params)
                    num_gt_ = num_gt
                    dataset_ = dataset
                    name_ = name
                    gt_ = gt
                    normalizer_ = normalizer

                del images, outputs

        scale_['scale_factor'] = scale_['scale_factor'].numpy()
        scale_['padding_u'] = scale_['padding_u'].numpy()
        scale_['padding_v'] = scale_['padding_v'].numpy()

        model_inference = {'image': image_.numpy(), 'heatmap': outputs_.numpy(),
                           'scale': scale_, 'num_gt': num_gt_, 'dataset': dataset_,
                           'name': name_, 'gt': gt_.numpy(), 'normalizer': normalizer_.numpy()}

        return model_inference

    def keypoint(self, infer):
        '''
        Scales the joints from heatmap to actual U, V on unscaled image
        Returns ground truth and predictions CSV
        :param infer: (dict) Dictionary returned by inference
        :return: (pd.DataFrame, pd.DataFrame) GT and Pred CSV
        '''

        image = infer['image']
        heatmap = infer['heatmap']
        scale = infer['scale']
        dataset = infer['dataset']
        name = infer['name']
        gt = infer['gt']
        normalizer = infer['normalizer']

        hm_uv_stack = []

        csv_columns = ['name', 'dataset', 'normalizer', 'joint', 'uv']

        gt_csv = []
        pred_csv = []

        # Iterate over all heatmaps to obtain predictions
        for i in range(image.shape[0]):

            heatmap_ = heatmap[i]

            gt_uv = gt[i]
            hm_uv = self.dataset_obj.estimate_uv(hm_array=heatmap_, pred_placeholder=-np.ones_like(gt_uv))
            hm_uv_stack.append(hm_uv)

            # Scaling the point ensures that the distance between gt and pred is same as the scale of normalization
            scale_factor = scale['scale_factor'][i]
            padding_u = scale['padding_u'][i]
            padding_v = scale['padding_v'][i]

            # Scaling ground truth
            gt_uv_correct = np.copy(gt_uv)
            hm_uv_correct = np.copy(hm_uv)

            gt_uv_correct[:, :, 1] -= padding_v
            gt_uv_correct[:, :, 0] -= padding_u
            gt_uv_correct /= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)

            # Scaling predictions
            hm_uv_correct[:, :, 1] -= padding_v
            hm_uv_correct[:, :, 0] -= padding_u
            hm_uv_correct /= np.array([scale_factor, scale_factor, 1]).reshape(1, 1, 3)

            assert gt_uv_correct.shape == hm_uv_correct.shape, "Mismatch in gt ({}) and prediction ({}) shape".format(
                gt_uv_correct.shape, hm_uv_correct.shape)

            # Iterate over joints
            for jnt in range(gt_uv_correct.shape[1]):
                gt_entry = {
                    'name': name[i],
                    'dataset': dataset[i],
                    'normalizer': normalizer[i],
                    'joint': self.ind_to_jnt[jnt],
                    'uv': gt_uv_correct[:, jnt, :].astype(np.float32)
                }

                pred_entry = {
                    'name': name[i],
                    'dataset': dataset[i],
                    'normalizer': normalizer[i],
                    'joint': self.ind_to_jnt[jnt],
                    'uv': hm_uv_correct[:, jnt, :].astype(np.float32)
                }

                gt_csv.append(gt_entry)
                pred_csv.append(pred_entry)

        # Visualize images on the 256, 256 images
        if self.viz:
            hm_uv = np.stack(hm_uv_stack, axis=0)
            self.visualize_predictions(image=image, name=name, dataset=dataset, gt=gt, pred=hm_uv)

        pred_csv = pd.DataFrame(pred_csv, columns=csv_columns)
        gt_csv = pd.DataFrame(gt_csv, columns=csv_columns)

        pred_csv.sort_values(by='dataset', ascending=True, inplace=True)
        gt_csv.sort_values(by='dataset', ascending=True, inplace=True)

        assert len(pred_csv.index) == len(gt_csv.index), "Mismatch in number of entries in pred and gt dataframes."

        pred_csv.to_csv(self.model_save_path.format("pred.csv"), index=False)
        gt_csv.to_csv(self.model_save_path.format("gt.csv"), index=False)
        logging.info('Pandas dataframe saved successfully.')

        return gt_csv, pred_csv

    def visualize_predictions(self, image=None, name=None, dataset=None, gt=None, pred=None):
        """
        Helper function to visualize predictions
        """
        dataset_viz = {}
        dataset_viz['img'] = image
        dataset_viz['name'] = name
        dataset_viz['split'] = np.ones(image.shape[0])
        dataset_viz['dataset'] = dataset
        dataset_viz['bbox_coords'] = np.zeros([image.shape[0], 4, 4])
        dataset_viz['num_persons'] = np.ones([image.shape[0], 1])
        dataset_viz['gt'] = gt
        dataset_viz['pred'] = pred

        dataset_viz = self.dataset_obj.recreate_images(gt=True, pred=True, external=True, ext_data=dataset_viz)
        visualize_image(dataset_viz, bbox=False, uv=True)

        return None


    def compute_metrics(self, gt_df=None, pred_df=None):
        '''
        Loads the ground truth and prediction CSVs into memory.
        Evaluates Precision, FPFN metrics for the prediction and stores them into memory.
        :return: None
        '''

        # Ensure that same datasets have been loaded
        assert all(pred_df['dataset'].unique() == gt_df['dataset'].unique()), \
            "Mismatch in dataset column for gt and pred"

        logging.info('Generating evaluation metrics for dataset:')
        # Iterate over unique datasets
        for dataset_ in gt_df['dataset'].unique():
            logging.info(str(dataset_))

            # Separate out images based on dataset
            pred_ = pred_df.loc[pred_df['dataset'] == dataset_]
            gt_ = gt_df.loc[gt_df['dataset'] == dataset_]

            # Compute scores
            pck_df = PercentageCorrectKeypoint(pred_df=pred_, gt_df=gt_, config=self.conf,
                                               jnts=list(self.ind_to_jnt.values()), data_name=dataset_)

            # Save the tables
            if dataset_ == 'mpii':
                metric_ = 'PCKh'
            else:
                metric_ = 'PCK'

            pck_df.to_csv(self.model_save_path.format("{}_{}.csv".format(metric_, dataset_)), index=False)

        print("Metrics computation completed.")

    def eval(self):
        '''
        Control flow to obtain predictions and corresponding metrics from Test()
        '''
        model_inference = self.inference()
        gt_csv, pred_csv = self.keypoint(model_inference)
        self.compute_metrics(gt_df=gt_csv, pred_df=pred_csv)


def config():
    conf = ParseConfig()

    if conf.success:
        logging.info('Successfully loaded config')
    else:
        logging.warn('Could not load configuration! Exiting.')
        exit()

    return conf


def load_models(conf=None, load_hg=True, load_learnloss=True, best_model=None, hg_param=None, model_dir=None):
    '''
    Initialize or load model(s): Hourglass, Learning Loss network

    :param conf: (Object of type ParseConfig) Contains the configuration for the experiment
    :param load_hg: (bool) Load Hourglass network
    :param load_learnloss: (bool) Load learning Loss network
    :param best_model: (bool) Load best model
    :param hg_param: (recheck type) Parameters for the Hourglass network
    :param model_dir: (string) Directory containing the model
    :return: (torch.nn x 2) Hourglass network, Learning Loss network
    '''

    epoch = conf.model_load_epoch

    # Learn Loss model - Load or train from scratch, will be defined even if not needed
    if load_learnloss:
        logging.info('Loading Learning Loss model from: ' + model_dir)
        learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
                                     conf.learning_loss_original)

        if best_model:
            if conf.resume_training:
                path_ = '_hg'  # *_hg will be the Learning Loss model at the epoch where HG gave best results
            else:
                path_ = ''  # best Learning Loss model
            learnloss_.load_state_dict(torch.load(
                model_dir
                + 'model_checkpoints/best_model_learnloss_{}'.format(conf.learning_loss_obj)
                + path_
                + '.pth', map_location='cpu'))
        else:
            learnloss_.load_state_dict(torch.load(model_dir + 'model_checkpoints/model_epoch_{}_learnloss.pth'.format(epoch), map_location='cpu'))
    else:
        logging.info('Defining the Learning Loss module. Training from scratch!')
        learnloss_ = LearnLossActive(conf.learning_loss_fc, conf.args['hourglass']['inp_dim'], 4,
                                     conf.learning_loss_original)

    # Hourglass MODEL - Load or train from scratch
    if load_hg:
        # Load model
        logging.info('Loading Hourglass model from: ' + model_dir)
        net_ = Hourglass(nstack=hg_param['nstack'], inp_dim=hg_param['inp_dim'], oup_dim=hg_param['oup_dim'],
                         bn=hg_param['bn'], increase=hg_param['increase'])

        if best_model:
            net_.load_state_dict(torch.load(os.path.join(model_dir, 'model_checkpoints/best_model.pth'), map_location='cpu'))
        else:
            net_.load_state_dict(torch.load(os.path.join(model_dir, 'model_checkpoints/model_epoch_{}.pth'.format(epoch)), map_location='cpu'))

        logging.info("Successfully loaded Model")

    else:
        # Define model and train from scratch
        logging.info('Defining the network - Stacked Hourglass. Training from scratch!')
        net_ = Hourglass(nstack=hg_param['nstack'], inp_dim=hg_param['inp_dim'], oup_dim=hg_param['oup_dim'],
                         bn=hg_param['bn'], increase=hg_param['increase'])

    # Multi-GPU / Single GPU
    logging.info("Using " + str(torch.cuda.device_count()) + " GPUs")
    net = net_
    learnloss = learnloss_

    if torch.cuda.device_count() > 1:
        # Hourglass net has cuda definitions inside __init__(), specify for learnloss
        learnloss.cuda(torch.device('cuda:1'))
    else:
        # Hourglass net has cuda definitions inside __init__(), specify for learnloss
        learnloss.cuda(torch.device('cuda:0'))
    logging.info('Successful: Model transferred to GPUs.')

    return net, learnloss


def define_hyperparams(conf, net, learnloss):
    '''
    Defines the hyperparameters of the experiment

    :param conf: (Object of type ParseConfig) Contains the configuration for the experiment
    :param net: (torch.nn) HG model
    :return: (dict) hyperparameter dictionary
    '''
    logging.info('Initializing the hyperparameters for the experiment.')
    hyperparameters = dict()
    hyperparameters['optimizer_config'] = {
                                           'lr': conf.lr,
                                           'weight_decay': conf.weight_decay
                                          }
    hyperparameters['loss_params'] = {'size_average': True}
    hyperparameters['num_epochs'] = conf.epochs
    hyperparameters['start_epoch'] = 0  # Used for resume training

    # Parameters declared to the optimizer
    if conf.train_learning_loss:
        logging.info('Parameters of Learning Loss and Hourglass networks passed to Optimizer.')
        params_list = [{'params': net.parameters()},
                       {'params': learnloss.parameters()}]
    else:
        logging.info('Parameters of Hourglass passed to Optimizer')
        params_list = [{'params': net.parameters()}]

    hyperparameters['optimizer'] = torch.optim.Adam(params_list, **hyperparameters['optimizer_config'])

    if conf.resume_training:
        logging.info('Loading optimizer state dictionary')
        if conf.best_model:
            optim_dict = torch.load(conf.model_load_path + 'model_checkpoints/optim_best_model.tar')

        else:
            assert type(conf.model_load_epoch) == int, "Load epoch for optimizer not specified"
            optim_dict = torch.load(conf.model_load_path + 'model_checkpoints/optim_epoch_{}.tar'.format(
                conf.model_load_epoch))

        # If the previous experiment used learn_loss, ensure the llal model is loaded, with the correct optimizer
        assert optim_dict['learn_loss'] == conf.model_load_learnloss, "Learning Loss model needed to resume training"

        hyperparameters['optimizer'].load_state_dict(optim_dict['optimizer_load_state_dict'])
        logging.info('Optimizer state loaded successfully.\n')

        logging.info('Optimizer and Training parameters:\n')
        for key in optim_dict:
            if key == 'optimizer_load_state_dict':
                logging.info('Param group length: {}'.format(len(optim_dict[key]['param_groups'])))
            else:
                logging.info('Key: {}\tValue: {}'.format(key, optim_dict[key]))

        logging.info('\n')

        if conf.resume_training:
            hyperparameters['start_epoch'] = optim_dict['epoch']

    hyperparameters['loss_fn'] = torch.nn.MSELoss(reduction='none')

    return hyperparameters


def debug_viz(data_obj):
    '''
    Small code snippet to visualize heatmaps
    :return:
    '''
    for i in range(30, 40):
        _, hm, _, _, _, _, _, _, _ = data_obj.__getitem__(i)
        fig, ax = plt.subplots(3, 5)
        for j in range(hm.shape[0]):
            ax[j // 5, j % 5].imshow(hm[j])
            ax[j // 5, j % 5].set_title('Joint: {}'.format(j))
        plt.show()


def main():
    '''

    :return:
    '''

    # Load configuration file
    conf  = config()
    args = conf.args

    # Loading MPII, LSPET
    logging.info('Loading MPII, LSPET, LSP')
    mpii_dict, max_persons_in_mpii = load_hp_dataset(mpii=True, conf=conf)
    lspet_dict = load_hp_dataset(lspet=True, conf=conf)
    lsp_dict = load_hp_dataset(lsp=True, conf=conf)

    args['mpii_params']['max_persons'] = max_persons_in_mpii

    # Defining the network
    hg_param = args['hourglass']
    network, learnloss = load_models(conf=conf, load_hg=conf.model_load_hg, load_learnloss=conf.model_load_learnloss,
                                     best_model=conf.best_model, hg_param=hg_param, model_dir=conf.model_load_path)

    # Defining the Active Learning library
    active_learning_obj = ActiveLearning(conf=conf, hg_network=network, learnloss_network=learnloss)

    # Defining DataLoader
    logging.info('Defining DataLoader.')
    dataset_lspet_lsp_mpii = Dataset_MPII_LSPET_LSP(mpii_dict=mpii_dict, lspet_dict=lspet_dict, lsp_dict=lsp_dict,
                                                    activelearning_obj=active_learning_obj, conf=conf,
                                                    getitem_dump=conf.model_save_path, **args)

    # Defining Hyperparameters
    hyperparameters = define_hyperparams(conf, network, learnloss)

    # Tensorboard
    writer = SummaryWriter(log_dir=os.path.join(conf.model_save_path[:-20], 'tensorboard'))

    if conf.train:
        train_obj = Train(network=network, learnloss=learnloss, hyperparameters=hyperparameters,
                          dataset_obj=dataset_lspet_lsp_mpii, conf=conf, tb_writer=writer)
        train_obj.train_model()

        # Reload the best model for metric evaluation
        network, learnloss = load_models(conf=conf, load_hg=True, load_learnloss=False, best_model=True,
                                         hg_param=hg_param, model_dir=conf.model_save_path[:-20])

    if conf.metric:
        metric_obj = Metric(network=network, dataset_obj=dataset_lspet_lsp_mpii, conf=conf)
        metric_obj.eval()

    #if args['misc']['viz']:
    #    visualize_image(dataset_lspet_lsp_mpii.recreate_images(gt=True, train=True), bbox=True)


if __name__ == "__main__":
    main()