import os
import torch
import torch.nn as nn
from network import Model
from tools import TripletLoss_WRT
from network.scheduler import WarmupMultiStepLR

class Base:
    def __init__(self, config):
        self.config = config
        self.pid_num = config.pid_num
        self.output_path = config.output_path
        self.save_model_path = os.path.join(self.output_path, 'models/')
        self.save_logs_path = os.path.join(self.output_path, 'logs/')

        self.learning_rate = config.learning_rate
        self.lr_times = config.lr_times
        self.weight_decay = config.weight_decay
        self.steps = config.steps
        self.num_epochs = config.total_train_epoch

        self.img_h = config.img_h
        self.img_w = config.img_w
        self.sequence_length = config.sequence_length

        self._init_device()
        self._init_model()
        self._init_criteria()
        self._init_optimizer()

    def _init_device(self):
        self.device = torch.device('cuda')

    def _init_model(self):
        self.model = Model(self.config,self.pid_num, self.img_h, self.img_w)
        self.model = self.model.to(self.device)


    def _init_criteria(self):
        self.id_criterion = nn.CrossEntropyLoss()
        self.triplet_criterion = TripletLoss_WRT()

    def _init_optimizer(self):
        params = []
        for key, value in self.model.named_parameters():
            if not value.requires_grad:
                continue
            lr = self.learning_rate
            weight_decay = self.weight_decay
            if "bias" in key:
                lr = self.learning_rate * self.lr_times
                weight_decay = self.weight_decay

            if "classifier" in key:
                lr = self.learning_rate * self.lr_times


            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        self.model_optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)


        self.lr_scheduler = WarmupMultiStepLR(
            self.model_optimizer,

            milestones=[10, 20, 30, 40, 50, 60],
            gamma=1.0/3,
            warmup_factor=0.005,
            warmup_iters=4,
            warmup_method='linear',
        )


    def save_model(self, save_epoch, is_best):
        if is_best:
            model_file_path = os.path.join(self.save_model_path, 'best_model.pth'.format(save_epoch))
            torch.save(self.model.state_dict(), model_file_path)

    def resume_model(self, resume_path):
        model_path = os.path.join(resume_path, 'best_model.pth')
        self.model.load_state_dict(torch.load(model_path), strict=False)
        print('Successfully resume shared_model from {}'.format(model_path))


    def set_train(self):
        self.model = self.model.train()
        self.training = True

    def set_eval(self):
        self.model = self.model.eval()
        self.training = False
