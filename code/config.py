import os
import yaml
import shutil
import logging


class ParseConfig(object):
    '''
    Stores the configurations as object
    '''

    def __init__(self):
        '''
        Loads the configurations as specified in the file configuration.yml
        '''

        self.success = True
        conf_yml = None

        try:
            f = open('configuration.yml', 'r')
            conf_yml = yaml.load(f)
            f.close()
        except FileNotFoundError:
            logging.warn('Could not find configuration.yml')
            self.success = False

        if self.success:

            self.experiment_name = conf_yml['experiment_name']

            # Creates unique model save path to avoid overwriting
            i_ = 1
            self.model_save_path = os.path.join(conf_yml['model_save'], self.experiment_name + '_' + str(i_), 'model_checkpoints/{}')
            while os.path.exists(self.model_save_path[:-3]):
                i_ += 1
                self.model_save_path = os.path.join(conf_yml['model_save'], self.experiment_name + '_' + str(i_), 'model_checkpoints/{}')
            logging.info('\nSaving the model at: ' + self.model_save_path[:-3])
            os.makedirs(self.model_save_path[:-3], exist_ok=True)

            # TODO: Copy code directory instead of individual files
            os.makedirs(os.path.join(self.model_save_path[:-20], 'code_files'), exist_ok=True)
            for file in conf_yml['files_to_copy']:
                try:
                    shutil.copyfile(src=file, dst=os.path.join(self.model_save_path[:-20], 'code_files', file))
                except FileNotFoundError:
                    logging.warn('Could not copy files!')
                    self.success = False

            logging.info("File(s) successfully saved at: ")
            logging.info(os.path.join(self.model_save_path[:-20], "code_files"))

        if self.success:

            self.train = conf_yml['train']
            self.metric = conf_yml['metric']
            self.model_load_hg = conf_yml['model_load_HG']
            self.learnloss_only = conf_yml['learnloss_only']
            self.model_load_learnloss = conf_yml['model_load_LearnLoss']
            self.model_load_path = conf_yml['model_load_path']
            self.resume_training = conf_yml['resume_training']
            self.model_load_epoch = conf_yml['load_epoch']
            self.best_model = conf_yml['best_model']

            self.epochs = conf_yml['epochs']
            self.lr = conf_yml['lr']
            self.weight_decay = conf_yml['weight_decay']
            self.batch_size = conf_yml['batch_size']
            self.num_hm = conf_yml['num_heatmap']
            self.hm_peak = conf_yml['args']['misc']['hm_peak']
            self.threshold = conf_yml['args']['misc']['threshold']
            self.n_stack = conf_yml['args']['hourglass']['nstack']

            self.train_learning_loss = conf_yml['args']['learning_loss_network']['train']
            self.learning_loss_margin = conf_yml['args']['learning_loss_network']['margin']
            self.learning_loss_warmup = conf_yml['args']['learning_loss_network']['warmup']
            self.learning_loss_fc = conf_yml['args']['learning_loss_network']['fc']
            self.learning_loss_original = conf_yml['args']['learning_loss_network']['original']
            self.learning_loss_obj = conf_yml['args']['learning_loss_network']['training_obj']

            self.args = conf_yml['args']
            self.mpii_newell_validation = conf_yml['args']['mpii_newell_validation']
            self.active_learning_params = conf_yml['active_learning']

            self.precached_mpii = conf_yml['precached_mpii']

            # Assertion checks if the code is run in MPII_only mode.
            if self.args['mpii_only']:
                assert conf_yml['num_heatmap'] == 16, "MPII has 16 joints, but heatmaps defined are: {}".format(
                    conf_yml['num_heatmap']
                )

                assert not self.args['mpii_params']['del_extra_jnts'], "del_extra_jnts is set to True"
                assert self.args['hourglass']['oup_dim'] == 16, "MPII has 16 joints, but heatmaps defined are: {}".format(
                    self.args['hourglass']['oup_dim']
                )

            else:
                assert conf_yml['num_heatmap'] == 14, "LSPET has 14 joints, but heatmaps defined are: {}".format(
                    conf_yml['num_heatmap'])

                assert self.args['hourglass']['oup_dim'] == 14, "LSPET has 14 joints, but heatmaps defined are: " \
                                                                "{}".format(self.args['hourglass']['oup_dim'])