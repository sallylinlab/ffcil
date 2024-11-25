from ntpath import join
import gc
import random
import heapq
import os.path as osp

import torch.nn as nn
import torchvision
import pandas as pd
import seaborn as sns
import tensorboardX as tbx
import matplotlib.pyplot as plt
from os import terminal_size

from copy import deepcopy
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, precision_score, f1_score, recall_score
from scipy.ndimage import gaussian_filter, map_coordinates

from .helper import *
from utils import *
from dataloader.data_utils import *
from .Network import MYNET
from models.loss_function import *
from models.base.fscil_trainer import FSCILTrainer as Trainer

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

import cv2
import csv

class FSCILTrainer(Trainer):
    def __init__(self, args):

        super().__init__(args)
        self.class27_names= ['I-Nothing',
                        'I-Others',
                        'E-M1-Al Residue',
                        'I-Dust',
                        'T-ITO1-Hole',
                        'T-M-Fiber',
                        'I-Sand Defect',
                        'I-Glass Scratch',
                        'E-AS-Defect',
                        'I-Oil Like',
                        'E-ITO1-Hole',
                        'P-ITO1-Residue',
                        'I-Laser Repair',
                        'P-AS-NO',
                        'T-ITO1-Residue',
                        'I-M2-Crack',
                        'E-M2-Residue',
                        'T-Brush Defect',   
                        'T-AS-Residue',
                        'T-AS-Particle Small',
                        'E-M2-PR Residue',
                        'T-M-Particle',
                        'P-M2-Residue',
                        'P-AS-Residue',
                        'P-M1-Residue',
                        'T-AS-SiN Hole',
                        'P-M2-Open']
        self.class_names= ['I-Nothing',
                        'I-Others',
                        'E-M1-Al Residue',
                        'I-Dust',
                        'T-ITO1-Hole',
                        'T-M2-Fiber',
                        'I-Sand Defect',
                        'I-Glass Scratch',
                        'E-AS-Residue',
                        'I-Oil Like',
                        'T-M1-Fiber',
                        'E-ITO1-Hole',
                        'P-ITO1-Residue',
                        'E-AS-BPADJ',
                        'I-Laser Repair',
                        'P-AS-NO',
                        'T-ITO1-Residue',
                        'I-M2-Crack',
                        'E-M2-Residue',
                        'T-Brush Defect',   
                        'T-AS-Residue',
                        'T-AS-Particle Small',
                        'E-M2-PR Residue',
                        'T-M2-Particle',
                        'P-M2-Residue',
                        'P-AS-Residue',
                        'P-M1-Residue',
                        'T-M1-Particle',
                        'P-M2-Short',
                        'T-AS-SiN Hole',
                        'P-AS-BPADJ',
                        'P-M2-Open',
                        'P-M1-Short']
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()
        self.f_loss = focal_loss(self.args)
        self.ring_loss = RingLoss()
        if args.epochs_res > 0:
            self.center_loss = CenterLoss(args.base_class+args.low_way, 512)
        else:
            self.center_loss = CenterLoss(args.episode_way+args.low_way, 512)

        self.acc_list = []
        self.ca = []
        self.class_accuracy = []
        self.class_fscore = []
        self.class_precision = []
        self.class_recall = []
        self.task_acc = []
        self.old_acc = []
        self.others = []
        for i in range(args.num_classes):
            self.ca.append([])
            self.class_fscore.append([])
            self.class_precision.append([])
            self.class_recall.append([])
            self.class_accuracy.append([])

        for i in range(args.sessions):
            self.task_acc.append([])

        pass

    def set_up_model(self):
        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir != None:  #
            print('Loading init parameters from: %s' % self.args.model_dir)

            if 'chris' in self.args.model_dir and self.args.epochs_base>0:
                self.best_model_dict = torch.load(self.args.model_dir)
                self.best_model_dict = {k[11:]:v for k,v in self.best_model_dict.items()}
                self.best_model_dict = {'conv1.weight' if k == 'conv1.0.weight' else k:v for k,v in self.best_model_dict.items()}
                self.best_model_dict = {'bn1.weight' if k == 'conv1.1.weight' else k:v for k,v in self.best_model_dict.items()}
                self.best_model_dict = {'bn1.bias' if k == 'conv1.1.bias' else k:v for k,v in self.best_model_dict.items()}
                self.best_model_dict = {'bn1.running_mean' if k == 'conv1.1.running_mean' else k:v for k,v in self.best_model_dict.items()}
                self.best_model_dict = {'bn1.running_var' if k == 'conv1.1.running_var' else k:v for k,v in self.best_model_dict.items()}
                try:
                    del self.best_model_dict['weight']
                    del self.best_model_dict['ier.weight']
                    del self.best_model_dict['conv1.1.num_batches_tracked']
                except Exception as e:
                    print(e)
            elif 'leo' in self.args.model_dir:
                self.best_model_dict = torch.load(self.args.model_dir)
            else:
                self.best_model_dict = torch.load(self.args.model_dir)['params']
        else:
            print('*********WARNINGl: NO INIT MODEL**********')
            # raise ValueError('You must initialize a pre-trained model')
            pass

    def update_param(self, model, pretrained_dict):
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        self.best_model_dict = deepcopy(self.model.state_dict())
        return model

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = self.get_base_dataloader_meta()
        else:
            trainset, trainloader, testloader = self.get_new_dataloader(session)
        return trainset, trainloader, testloader

    def get_base_dataloader_meta(self):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(0 + 1) + '.txt'
        class_index = np.arange(self.args.base_class)
        if self.args.dataset == 'cifar100':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=True,
                                                  index=class_index, base_sess=True)
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_index, base_sess=True)

        if self.args.dataset == 'cub200':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'auo':
            trainset = self.args.Dataset.AUO(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.AUO(root=self.args.dataroot, train=False, index=class_index)
            
        # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        sampler = CategoriesSampler(trainset.targets, self.args.train_episode, self.args.episode_way,
                                    self.args.episode_shot + self.args.episode_query)

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=self.args.num_workers,
                                                  pin_memory=True)
        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=self.args.num_workers, pin_memory=True)
        
        return trainset, trainloader, testloader

    def get_new_dataloader(self, session):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(session + 1) + '.txt'
        if self.args.dataset == 'cifar100':
            class_index = open(txt_path).read().splitlines()
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=False,
                                                  index=class_index, base_sess=False)
        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True,
                                                index_path=txt_path)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        if self.args.dataset == 'auo':
            trainset = self.args.Dataset.AUO(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)

        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=self.args.num_workers, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=self.args.num_workers, pin_memory=True)

        class_new = self.get_session_classes(session)

        if self.args.dataset == 'cifar100':
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_new, base_sess=False)
        if self.args.dataset == 'cub200':
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_new)
        if self.args.dataset == 'mini_imagenet':
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_new)
        if self.args.dataset == 'auo':
            testset = self.args.Dataset.AUO(root=self.args.dataroot, train=False, index=class_new)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=self.args.num_workers, pin_memory=True)

        return trainset, trainloader, testloader

    def get_low_dataloader(self):
        trainset = self.args.Dataset.AUO(root=self.args.dataroot, train=False, index=np.arange(25), extra=True)
        sampler = CategoriesSampler(trainset.targets, self.args.train_episode, self.args.low_way,
                            self.args.low_shot + self.args.episode_query)

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=self.args.num_workers,
                                                  pin_memory=True)

        return trainloader

    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def replace_to_rotate(self, proto_tmp, query_tmp):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=(0.5 , 1.2), contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.1, 0.1))
        ])


        # random choose rotate or color augmentation
        for i in range(self.args.low_way):
            r = random.random()
            if self.args.pseudo_mode == 'rotate':
                r = 1
            elif self.args.pseudo_mode == 'color':
                r = 0

            if r>0.5:
              # random choose rotate degree
              rot_list = [90, 180, 270]
              sel_rot = random.choice(rot_list)
              if sel_rot == 90:  # rotate 90 degree
                  # print('rotate 90 degree')
                  proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
                  query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(2)
              elif sel_rot == 180:  # rotate 180 degree
                  # print('rotate 180 degree')
                  proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].flip(2).flip(3)
                  query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].flip(2).flip(3)
              elif sel_rot == 270:  # rotate 270 degree
                  # print('rotate 270 degree')
                  proto_tmp[i::self.args.low_way] = proto_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
                  query_tmp[i::self.args.low_way] = query_tmp[i::self.args.low_way].transpose(2, 3).flip(3)
            else:
                # color augmentation
                p_q = torch.cat((proto_tmp[i::self.args.low_way].clone(), query_tmp[i::self.args.low_way].clone()),0)
                p_q = transform(p_q)
                proto_tmp[i::self.args.low_way] = p_q[:self.args.low_shot]
                query_tmp[i::self.args.low_way] = p_q[self.args.low_shot:]
                
        return proto_tmp, query_tmp
    # FGSM attack code
    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        # Return the perturbed image
        return perturbed_image

    # restores the tensors to their original scale
    def denorm(self, batch, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Convert a batch of tensors to their original scale.

        Args:
            batch (torch.Tensor): Batch of normalized tensors.
            mean (torch.Tensor or list): Mean used for normalization.
            std (torch.Tensor or list): Standard deviation used for normalization.

        Returns:
            torch.Tensor: batch of tensors without normalization applied to them.
        """
        if isinstance(mean, list):
            mean = torch.tensor(mean).cuda()
        if isinstance(std, list):
            std = torch.tensor(std).cuda()

        return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)    

    def make_pseudo_classes(self, proto, c, shot, masks):
        # 1 5 512
        pseudo = proto.clone().detach()
        pseudo_temp = proto.clone().detach()
        # for i in range(len(c)):
        #     print(f'{i+5}:{c[i]}')

        for j in range(shot):
            for i in range(self.args.low_way):
                # p1 = self.topk_mask(pseudo_temp[j][c[i][0]].clone())
                # p2 = self.topk_mask(pseudo_temp[j][c[i][1]].clone())
                # p1 = pseudo_temp[j][c[i][0]].clone().mul(masks[c[i][0]])
                # p2 = pseudo_temp[j][c[i][1]].clone().mul(masks[c[i][1]])
                p1 = pseudo_temp[j][c[i][0]].clone()
                p2 = pseudo_temp[j][c[i][1]].clone()

                # avg
                t = torch.stack((p1,p2), 0)
                pseudo[j][i] = t.mean(0)
                # print(t.mean(0))
                # only mask
                # p1 = pseudo_temp[j][i].clone().mul(masks[i])
                # pseudo[j][i] = p1

        return pseudo

    def topk_mask(self, feature, k=5):
        
        # topk = heapq.nlargest(k, feature)
        # for i in range(len(feature)):
        #     if feature[i] in topk:
        #         feature[i] = torch.zeros(1)[0].cuda()
        for i in range(k):
            index = torch.argmax(feature)
            feature[index] = torch.tensor(0).cuda()

        return feature

    def make_noise_etransform(self, images):
        
        # images shape = amount, channels, height, width
        # make images to amount, height, width, channels
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        # images = np.moveaxis(images.numpy(), -1, 1)

        amount=0.02
        alpha=200
        sigma=9
        alpha_affine=9
        random_state = np.random.RandomState(None)

        pseudo_images = []

        for image in images:

            # height, width, channels = image.shape
            # num_pixels = int(amount * height * width)

            # for _ in range(num_pixels):
            #     x = random.randint(0, width - 1)
            #     y = random.randint(0, height - 1)
            #     value = random.randint(0, 1) * 255  # 0 for black (pepper), 255 for white (salt)
            #     image[y, x] = [value, value, value]

            shape = image.shape
            shape_size = shape[:2]

            # Random affine
            center_square = np.float32(shape_size) // 2
            square_size = min(shape_size) // 3
            pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
            pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
            M = cv2.getAffineTransform(pts1, pts2)
            image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
            dz = np.zeros_like(dx)

            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
            indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
            pseudo_image = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
            pseudo_images.append(pseudo_image)
            # pseudo_images.append(image)
        
        return torch.from_numpy(np.array(pseudo_images)).permute(0, 3, 1, 2).cuda()

    def get_elastic_transform_pseudo_classes(self, proto, query):

        # transform = transforms.Compose([
        #     torchvision.transforms.ToTensor(),
        #     torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
        proto_clone = proto.detach().clone()
        query_clone = query.detach().clone()

        proto_clone = self.denorm(proto_clone)
        query_clone = self.denorm(query_clone)

        proto_tmp = self.make_noise_etransform(proto_clone)
        query_tmp = self.make_noise_etransform(query_clone)

        proto_tmp_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(proto_tmp)
        query_tmp_norm = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(query_tmp)

        return proto_tmp_norm, query_tmp_norm



    def get_rotate_pseudo_classes(self, proto, query):

        args = self.args
        proto = proto.detach()
        query = query.detach()

        # random choose rotate degree
        # sample low_way data
        proto_tmp = deepcopy(
            proto.reshape(args.episode_shot, args.episode_way, proto.shape[1], proto.shape[2], proto.shape[3])[
            :args.low_shot,
            :args.low_way, :, :, :].flatten(0, 1))
        query_tmp = deepcopy(
            query.reshape(args.episode_query, args.episode_way, query.shape[1], query.shape[2], query.shape[3])[
            :,
            :args.low_way, :, :, :].flatten(0, 1))
        
        # random choose rotate degree
        proto_tmp, query_tmp = self.replace_to_rotate(proto_tmp, query_tmp)

        return proto_tmp, query_tmp

    def get_triplet_loss_data(self, query):

        args = self.args
        positive = torch.tensor([]).cuda()
        negative = torch.tensor([]).cuda()

        for i in range(args.episode_way):
            row_index = random.randint(0, args.episode_way -1)
            row_index_2 = random.randint(args.episode_way, len(query) - 1)
            row = query[row_index]
            n_index = random.randint(0, args.episode_way-1)

            positive = torch.cat((positive, torch.unsqueeze(query[row_index][i], 0)))
            negative = torch.cat((negative, torch.unsqueeze(query[row_index_2][n_index], 0)))

        for i in range(args.low_way):
            row_index = random.randint(0, args.episode_way -1)
            row_index_2 = random.randint(args.episode_way, len(query) - 1)
            row = query[row_index]
            n_index = random.randint(0, args.episode_way-1)

            positive = torch.cat((positive, torch.unsqueeze(query[row_index_2][i], 0)))
            negative = torch.cat((negative, torch.unsqueeze(query[row_index][n_index], 0)))

        return positive, negative

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                     {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)

        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_optimizer_res(self):
        optimizer = torch.optim.SGD([{'params': self.model.module.encoder.parameters(), 'lr': self.args.lr_base},
                                    {'params': self.model.module.slf_attn.parameters(), 'lr': self.args.lrg}],
                                    momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        # optimizer = torch.optim.SGD(self.model.module.encoder.parameters(), lr=self.args.lr_base,
        #                             momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        optimizer_center = torch.optim.SGD(self.center_loss.parameters(), lr=0.01, momentum=0.9)
        # params = list(self.model.module.encoder.parameters()) + list(self.model.module.slf_attn.parameters())
        # optimizer = torch.optim.SGD(self.model.module.slf_attn.parameters(), lr=self.args.lrg,
        #                             momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        scheduler_center = torch.optim.lr_scheduler.StepLR(optimizer_center, step_size=self.args.step, gamma=self.args.gamma)
        return [optimizer, optimizer_center], [scheduler, scheduler_center]
        # return optimizer, scheduler

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model = self.update_param(self.model, self.best_model_dict)
            self.best_model_dict = deepcopy(self.model.state_dict())

            if session == 0:  # load base class train img label
                s_t = time.strftime("%H%M", time.localtime())
                # self.get_gcam()
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                if args.extra_class:
                    lowloader = self.get_low_dataloader()
                else:
                    lowloader = None

                # self.classifier_train(self.model, testloader, args)

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    self.model.eval()
                    tl, ta = self.base_train(self.model, trainloader, lowloader, optimizer, scheduler, epoch, args)

                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.model.module.mode = 'avg_cos'

                    if args.set_no_val: # set no validation
                        
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        tsl, tsa = self.test(self.model, testloader, args, session)
                        self.trlog['test_loss'].append(tsl)
                        self.trlog['test_acc'].append(tsa)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                        result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                                epoch, lrc, tl, ta, tsl, tsa))
                        
                    else:
                        # take the last session's testloader for validation
                        vl, va = self.validation()

                        # save better model
                        if (va * 100) >= self.trlog['max_acc'][session]:
                            self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                            self.trlog['max_acc_epoch'] = epoch
                            save_model_dir = os.path.join(args.save_path, 'session' + str(session) + f'_{s_t}.pth')
                            torch.save(dict(params=self.model.state_dict()), save_model_dir)
                            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                            self.best_model_dict = deepcopy(self.model.state_dict())
                            print('********A better model is found!!**********')
                            print('Saving model to :%s' % save_model_dir)
                        print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                        self.trlog['max_acc'][session]))
                        self.trlog['val_loss'].append(vl)
                        self.trlog['val_acc'].append(va)
                        lrc = scheduler.get_last_lr()[0]
                        print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))
                        result_list.append(
                            'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                                epoch, lrc, tl, ta, vl, va))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)

                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                
                # res training
                print('start res train')
                optimizer, scheduler = self.get_optimizer_res()
                result_list.append('\nres training log')

                for epoch in range(args.epochs_res):
                    start_time = time.time()
                    # train base sess
                    self.model.train()
                    self.model.module.encoder.eval()
                    self.model.module.slf_attn.train()
                    tl, ta = self.res_train(self.model, trainloader, lowloader, optimizer, scheduler, epoch, args)
                    self.model.module.slf_attn.eval()
                    self.model.eval()
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    self.model.module.mode = 'avg_cos'

                    vl, va = self.validation()
                    # save better model
                    if (va * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (va * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, str(session) + f'_res{s_t}.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                    print('best epoch {}, best val acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                    self.trlog['max_acc'][session]))

                    lrc = scheduler[0].get_last_lr()[0]
                    print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                        epoch, lrc, tl, ta, vl, va))
                    result_list.append(
                        'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))
                    print('still need around %.2f mins to finish' % (
                                  (time.time() - start_time) * (args.epochs_res - epoch) / 60))
                    # scheduler.step()
                    scheduler[0].step()
                    scheduler[1].step()




                # self.classifier_train(self.model, trainloader, testloader, args)


                print('session0 testing')
                # always replace fc with avg mean
                self.model.load_state_dict(self.best_model_dict)
                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                # best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                # if args.epochs_base != 0:
                #     torch.save(dict(params=self.model.state_dict()), best_model_dir)

                self.model.module.mode = 'avg_cos'
                tsl, tsa = self.test(self.model, testloader, args, session, getc_acc=True)

                # print('The test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))
                result_list.append(self.trlog['session_acc'][session])

            else:  # incremental learning sessions
                # print("training session: [%d]" % session)
                self.model.load_state_dict(self.best_model_dict)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                # validset = self.args.Dataset.AUO(root=self.args.dataroot, train=False, index=self.get_session_classes(0), valid=True)

                # validloader = torch.utils.data.DataLoader(dataset=validset, batch_size=64, shuffle=False,
                #                                         num_workers=self.args.num_workers, pin_memory=True)
                # self.get_tsne_(self.model, validloader)

                tsl, tsa = self.test(self.model, testloader, args, session, getc_acc=True, new_stage=True)

                # save better model
                # save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                # torch.save(dict(params=self.model.state_dict()), save_model_dir)
                # self.best_model_dict = deepcopy(self.model.state_dict())
                # print('Saving model to :%s' % save_model_dir)
                # print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append(self.trlog['session_acc'][session])


        print('acc: ', self.trlog['max_acc'])
        print('fscore: ', self.trlog['max_fscore'])

        last_acc = []
        base_acc = []
        last_precision = []
        for i in range(args.num_classes):
            last_acc.append(self.ca[i][-1])
            base_acc.append(self.ca[i][0])
            last_precision.append(self.class_precision[i][-1])
        
        # print(base_acc[:17])
        
        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        result_list.append('Best epoch:%d' % self.trlog['max_acc_epoch'])
        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append('session acc:{}'.format(self.trlog['max_acc']))
        result_list.append('session fscore:{}'.format(self.trlog['max_fscore']))
        result_list.append('other list:{}'.format(self.others))
        result_list.append('all acc: {:.2f}; base acc: {:.2f}; new acc: {:.2f}'.format(get_lavg(last_acc), get_lavg(last_acc[:args.base_class]), get_lavg(last_acc[args.base_class:])))
        result_list.append('all prec: {:.2f}; base prec: {:.2f}; new prec: {:.2f}\n'.format(get_lavg(last_precision), get_lavg(last_precision[:args.base_class]), get_lavg(last_precision[args.base_class:])))
        result_list.append('Last Session Class Accuracy: {}\n'.format(last_acc))

        result_list.append('Class Accuracy:')
        for i in range(args.num_classes):
            class_acc =''
            for j in self.ca[i]:
                class_acc=class_acc+str(j)+','
            result_list.append(class_acc)

        result_list.append('\nClass Recall:')
        for i in range(args.num_classes):
            class_acc =''
            for j in self.class_recall[i]:
                class_acc=class_acc+str(j)+','
            result_list.append(class_acc)

        result_list.append('\nClass Precision:')
        for i in range(args.num_classes):
            class_acc =''
            for j in self.class_precision[i]:
                class_acc=class_acc+str(j)+','
            result_list.append(class_acc)

        result_list.append('\nClass Fscore:')
        for i in range(args.num_classes):
            class_acc =''
            for j in self.class_fscore[i]:
                class_acc=class_acc+str(j)+','
            result_list.append(class_acc)

        for i in range(args.sessions):
            result_list.append('task{}:{}'.format(i+1, self.task_acc[i]))
        result_list.append('\nold task acc:{}'.format(self.old_acc))

        print(f'new class avg acc: {get_lavg(last_acc[args.base_class:])}')
        print('Best epoch:', self.trlog['max_acc_epoch'])
        print('session acc: ', self.trlog['max_acc'])
        print('session fscore: ', self.trlog['max_fscore'])
        print('\nTotal time used %.2f mins' % total_time)
        save_list_to_txt(os.path.join(args.save_path, f'results{s_t}.txt'), result_list)
        # for i in range(self.args.num_classes):
        #     print('class{}: {}'.format(i, self.ca[i]))

    def validation(self):
        with torch.no_grad():
            model = self.model

            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)
                if self.args.dataset == 'auo':
                    validset = self.args.Dataset.AUO(root=self.args.dataroot, train=False, index=self.get_session_classes(session), valid=True)

                    validloader = torch.utils.data.DataLoader(dataset=validset, batch_size=128, shuffle=False,
                                                            num_workers=self.args.num_workers, pin_memory=True)
                else:
                    validloader = testloader
                trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va = self.test(model, validloader, self.args, session)

        return vl, va

    def base_train(self, model, trainloader, low_trainloader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()

        tqdm_gen = tqdm(trainloader)
        
        # pseudo label
        label = torch.arange(args.episode_way + args.low_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)
        
        train_proto = list(np.arange(args.episode_way))
        
        low_loader = iter(low_trainloader)

        for i, batch in enumerate(tqdm_gen, 1):
            
            pseudo_index = []
            masks = []

            while len(pseudo_index)<args.low_way:
                t = sorted(random.sample(train_proto, 2))
                if t not in pseudo_index:
                    pseudo_index.append(t)
                    
            for i in range(args.episode_way):         
                mask = torch.ones(512)
                mask[:25] = torch.zeros(25)
                mask  = mask[torch.randperm(mask.size()[0])]
                masks.append(mask.cuda())
            
            data, true_label = [_.cuda() for _ in batch]

            k = args.episode_way * args.episode_shot
            model.module.mode = 'encoder'

            if args.extra_class:
                try:
                    low_data, low_label = next(low_loader)
                except StopIteration:
                    low_loader = iter(low_trainloader)
                    low_data, low_label = next(low_loader)

                low_data = low_data.cuda()
                low_data = model(low_data)
                k2 = args.low_way * args.low_shot
                proto_tmp, query_tmp = low_data[:k2], low_data[k2:]

            else:
                proto, query = data[:k], data[k:]
                proto_tmp, query_tmp = self.get_rotate_pseudo_classes(proto, query)
                proto_tmp = model(proto_tmp)
                query_tmp = model(query_tmp)


            data = model(data)
            proto, query = data[:k], data[k:]

            # 5w1s [5,512] to [1,5,512]
            # 5w5s [25,512] to [5,5,512]
            # qurey same between before and after .view [10,5,512]
            proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
            query = query.view(args.episode_query, args.episode_way, query.shape[-1])

            proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
            query_tmp = query_tmp.view(args.episode_query, args.low_way, query.shape[-1])
            
            # proto_tmp = self.make_pseudo_classes(proto, pseudo_index, args.episode_shot, masks)
            # query_tmp = self.make_pseudo_classes(query, pseudo_index, args.episode_query, masks)

            # self.get_tsne(proto, query, proto_tmp, query_tmp, 'stage2')
            
            # get class prototype, mean(0) 0=index of shot number 
            proto = proto.mean(0).unsqueeze(0)
            proto_tmp = proto_tmp.mean(0).unsqueeze(0)

            proto = torch.cat([proto, proto_tmp], dim=1)
            query = torch.cat([query, query_tmp], dim=1)

            proto = proto.unsqueeze(0)
            query = query.unsqueeze(0)

            logits = model.module._forward(proto, query)

            # total_loss = self.f_loss(logits, label)
            total_loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            lrc = scheduler.get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        tl = tl.item()
        ta = ta.item()
        return tl, ta

    def res_train(self, model, trainloader, low_trainloader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()
        tqdm_gen = tqdm(trainloader)

        label = torch.arange(start=args.base_class, end=args.base_class+args.low_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)

        triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)

        if args.extra_class:
            low_loader = iter(low_trainloader)
        
        for i, batch in enumerate(tqdm_gen, 1):
            
            data, true_label = [_.cuda() for _ in batch]
            data.requires_grad = True
            k = args.episode_way * args.episode_shot
            true_label = torch.cat((true_label, label), dim=0)[k:]
            model.module.mode = 'encoder'
            pseudo_index = []

            train_proto = list(np.arange(args.episode_way))
            while len(pseudo_index)<args.low_way:
                t = sorted(random.sample(train_proto, 2))
                if t not in pseudo_index:
                    pseudo_index.append(t)
                    

            if args.extra_class:
                try:
                    low_data, low_label = next(low_loader)
                except StopIteration:
                    low_loader = iter(low_trainloader)
                    low_data, low_label = next(low_loader)
                low_data = low_data.cuda()
                low_data = model(low_data)
                k2 = args.low_way * args.low_shot
                proto_tmp, query_tmp = low_data[:k2], low_data[k2:]

            else:
                proto, query = data[:k], data[k:]
                proto_tmp, query_tmp = self.get_rotate_pseudo_classes(proto, query)
                # proto_tmp, query_tmp = self.get_elastic_transform_pseudo_classes(proto, query)
                proto_tmp = model(proto_tmp)
                query_tmp = model(query_tmp)
            torch.cuda.empty_cache()

            k2 = args.episode_way * args.episode_query
            proto = data[:k]
            query = data[k:]

            proto = model(proto)
            query = model(query)

            # 5w1s [5,512] to [1,5,512]
            # 5w5s [25,512] to [5,5,512]
            proto = proto.view(args.episode_shot, args.episode_way, proto.shape[-1])
            proto_tmp = proto_tmp.view(args.low_shot, args.low_way, proto.shape[-1])
            # get class prototype, mean(0) 0=index of shot number 
            proto = proto.mean(0).unsqueeze(0).unsqueeze(0)
            proto_tmp = proto_tmp.mean(0).unsqueeze(0).unsqueeze(0)
            #p 1,1,e_way,512  #q 1,1,q_shot,512
            query = query.unsqueeze(0).unsqueeze(0)
            query_tmp = query_tmp.unsqueeze(0).unsqueeze(0)

            protos = model.module.fc.weight[:args.base_class, :].clone().detach().unsqueeze(0).unsqueeze(0)

            protos = torch.cat([protos, proto_tmp], dim=2)
            query = torch.cat([query, query_tmp], dim=2)
            logits = model.module._forward(protos, query)

            # anchor = torch.cat([proto, proto_tmp], dim=2).view(args.episode_way*2, proto.shape[-1])
            # query_trip = query.view(args.episode_query*2, args.episode_way, query.shape[-1])
            # positive, negative = self.get_triplet_loss_data(query_trip)

            # t_loss = triplet_loss(anchor, positive, negative)
            # gce_loss = GCE(logits, true_label)


            total_loss = F.cross_entropy(logits, true_label)
            # total_loss = F.mse_loss(logits, F.one_hot(true_label, num_classes = args.base_class+args.low_way).float())
            acc = count_acc(logits, true_label)

            lrc = scheduler[0].get_last_lr()[0]
            tqdm_gen.set_description(
                'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
            tl.add(total_loss.item())
            ta.add(acc)

            # optimizer.zero_grad()
            optimizer[0].zero_grad()
            optimizer[1].zero_grad()
            total_loss.backward()

            optimizer[0].step()
            optimizer[1].step()

        tl = tl.item()
        ta = ta.item()
        # del protos
        # del query
        # del data
        # gc.collect()
        # torch.cuda.empty_cache()
        return tl, ta

    def classifier_train(self, model, testloader, args):  

        optimizer = torch.optim.Adam([{'params': self.model.module.encoder.parameters()},
                                    {'params': self.model.module.classifier.parameters()}], weight_decay=self.args.decay)
        optimizer = torch.optim.Adam(self.model.module.classifier.parameters(), weight_decay=self.args.decay)
        train_set = args.Dataset.AUO(root=args.dataroot, train=True,
                                                    index=np.arange(args.base_class), base_sess=True)
        trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size_base, shuffle=True,
                                                num_workers=args.num_workers, pin_memory=True)

        max_acc = 0

        for epoch in range(20):
            model.train()
            tl = Averager()
            ta = Averager()
            tqdm_gen = tqdm(trainloader)
            for i, batch in enumerate(tqdm_gen, 1):
                data, true_label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                data = model(data)
                #p 1,1,e_way,512  #q 1,1,q_shot,512
                query = data.unsqueeze(0).unsqueeze(0)
                protos = model.module.fc.weight[:args.base_class, :].clone().detach().unsqueeze(0).unsqueeze(0)

                logits = model.module.forward_(protos, query)
                total_loss = F.mse_loss(logits, F.one_hot(true_label, num_classes = args.base_class).float())
                acc = count_acc(logits, true_label)

                lrc = optimizer.param_groups[0]["lr"]
                tqdm_gen.set_description(
                    'Session 0, epo {}, lrc={:.4f},total loss={:.4f} acc={:.4f}'.format(epoch, lrc, total_loss.item(), acc))
                tl.add(total_loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

            tl = tl.item()
            ta = ta.item()

            model = model.eval()
            # self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
            vl = Averager()
            va = Averager()

            with torch.no_grad():
                for i, batch in enumerate(testloader, 1):
                    data, test_label = [_.cuda() for _ in batch]
                    model.module.mode = 'encoder'
                    query = model(data)
                    query = query.unsqueeze(0).unsqueeze(0)
                    proto = model.module.fc.weight[:args.base_class, :].detach()
                    proto = proto.unsqueeze(0).unsqueeze(0)
                    logits = model.module.forward_(proto, query)

                    loss = F.cross_entropy(logits, test_label)
                    acc = count_acc(logits, test_label)

                    vl.add(loss.item())
                    va.add(acc)

                vl = vl.item()
                va = va.item()
            # save better model
            if (va * 100) >= max_acc:
                max_acc = va * 100
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('********A better model is found!!**********')
            print('best val acc={:.3f}'.format(max_acc))

            lrc = optimizer.param_groups[0]["lr"]
            print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                epoch, lrc, tl, ta, vl, va))

            self.model.load_state_dict(self.best_model_dict)
            save_model_dir = os.path.join(args.save_path, 'classifier.pth')
            torch.save(dict(params=self.model.state_dict()), save_model_dir)

    def test(self, model, testloader, args, session, getc_acc=False, new_stage=False):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        class_correct = [0]*args.num_classes
        class_data_size = [0]*args.num_classes
        task_correct = [0]*args.sessions
        task_data_size = [0]*args.sessions
        
        pred_list = []
        true_list = []
        num_others = 0

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                model.module.mode = 'encoder'
                query = model(data)
                query = query.unsqueeze(0).unsqueeze(0)

                proto = model.module.fc.weight[:test_class, :].detach()
                # if proto.shape[0]==33:
                #     writer = tbx.SummaryWriter('runs/rotate_all')
                #     writer.add_embedding(proto, metadata=np.arange(proto.shape[0]))
                #     writer.close()
                #     print('tsne done')
                #     input()

                proto = proto.unsqueeze(0).unsqueeze(0)
                logits = model.module._forward(proto, query)
                # if getc_acc:
                #     logits = model.module.forward_(proto, query)
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                p = torch.argmax(logits, dim=1).cpu().numpy()
                pred_top5 = torch.topk(logits, 5, dim=1)[1].cpu().numpy()
                # p =[]

                # for j in range(len(test_label)):
                #     top5 = []
                #     for k in range(5):
                #         top5.append(torch.argmax(logits[j]).cpu().numpy())
                #         logits[j][top5[k]] = -1
                #     if test_label[j] not in top5:
                #         p.append(top5[0])
                #     else:
                #         p.append(test_label[j].cpu().numpy())


                # for i in range(len(aa[0])):
                #     if aa[0][i]<0.5:
                #         p[i] = 27
                #         num_others += 1

                t = test_label.cpu().numpy()
                pred_list.extend(p)
                true_list.extend(t)
                
                for i in range(len(t)):
                    # get class acc
                    class_data_size[t[i]]+=1
                    if p[i]==t[i]:
                        class_correct[t[i]]+=1
                    # get task acc
                    for j in range(args.sessions):
                        if t[i]<args.base_class+args.way*j:
                            task_data_size[j]+=1
                            if p[i]==t[i]:
                                task_correct[j]+=1
                            break       


                vl.add(loss.item())
                va.add(acc)

            vl = vl.item()
            va = va.item()
            fs = f1_score(true_list, pred_list, average='macro')

        if getc_acc:
            all_correct = 0
            all_size = 0
            new_correct = 0
            new_size = 0

            recall = recall_score(true_list, pred_list, average=None)
            precision = precision_score(true_list, pred_list, average=None)
            fscore = f1_score(true_list, pred_list, average=None)

            for i in range(len(class_correct)):
                if class_data_size[i]==0:
                    self.ca[i].append(0)
                    self.class_recall[i].append(0)
                    self.class_precision[i].append(0)
                    self.class_fscore[i].append(0)
                    continue
                self.ca[i].append(round(class_correct[i]/class_data_size[i]*100, 2))
                self.class_recall[i].append(round(recall[i]*100, 2))
                self.class_precision[i].append(round(precision[i]*100, 2))
                self.class_fscore[i].append(round(fscore[i]*100, 2))

                all_correct+=class_correct[i]
                all_size+=class_data_size[i]
                if i >= args.base_class-1:
                    new_correct+=class_correct[i]
                    new_size+=class_data_size[i]

            for i in range(args.sessions):
                if task_data_size[i]==0:
                    continue
                self.task_acc[i].append(round(task_correct[i]/task_data_size[i]*100, 3))
            
            if session != 0:
                self.old_acc.append(round((all_correct-task_correct[session])/(all_size-task_data_size[session])*100, 3))

            self.trlog['max_fscore'][session] = float('%.3f' % (fs * 100))
            self.trlog['max_acc'][session] = round((all_correct/all_size)*100, 3)
            # print(f'------{num_others}------')
            self.others.append(num_others)

            # print(f'session:{session}, all_acc={all_correct/all_size}, base_acc={(all_correct-new_correct)/(all_size-new_size)}, new_acc={new_correct/new_size}')
            print(f'session:{session}, all_acc={get_lavg(recall)*100:.2f}, base_acc={get_lavg(recall[:args.base_class])*100:.2f}, new_acc={get_lavg(recall[args.base_class:])*100:.2f}')
            self.trlog['session_acc'].append(f'session:{session}, all_acc={get_lavg(recall)*100:.2f}, base_acc={get_lavg(recall[:args.base_class])*100:.2f}, new_acc={get_lavg(recall[args.base_class:])*100:.2f}')

                
        # if session == self.args.sessions-1 and new_stage:

        #     cf_matrix = confusion_matrix(true_list, pred_list)
        #     df_cm = pd.DataFrame(cf_matrix, self.class27_names, self.class27_names) 
        #     plt.figure(figsize = (18,12))
        #     ax = sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')

        #     label_x = ax.get_xticklabels()
        #     plt.setp(label_x, rotation=45, horizontalalignment='right')
            
        #     plt.ylabel('True label')
        #     plt.xlabel('Predicted label')
        #     pretrain_name = self.args.model_dir.split('/')[1].split('.')[0]
        #     plt.savefig(self.args.save_path + f'/{pretrain_name}_cm.png')

        #     cn = []
        #     for i in range(len(pred_list)):
        #         cn.append([self.class27_names[true_list[i]], self.class27_names[pred_list[i]]])
        #     with open(self.args.save_path + f'/{pretrain_name}.csv', 'w') as csvfile:
        #         writer = csv.writer(csvfile)
        #         for x in cn:
        #             writer.writerow(x)


        return vl, va

    def get_tsne(self, proto, query, proto_tmp, query_tmp, pseudo_type):
        labels = np.arange(self.args.episode_way+self.args.low_way).repeat(11)
        # labels = np.arange().repeat(11)
        features = None
        for i in range(self.args.episode_way):

                if features is None:
                    features = proto[0][i]
                    features = features.view(1,512)
                else:
                    features = torch.cat((features, proto[0][i].view(1,512)))
                for j in range(self.args.episode_query):
                    features = torch.cat((features, query[j][i].view(1,512)))
        for i in range(self.args.low_way):

                features = torch.cat((features, proto_tmp[0][i].view(1,512)))
                for j in range(self.args.episode_query):
                    features = torch.cat((features, query_tmp[j][i].view(1,512)))

        writer = tbx.SummaryWriter(f'runs/{pseudo_type}')
        features = features.cpu().detach().numpy()
        writer.add_embedding(features, metadata=labels)
        writer.close()

        print('tsne done')
        input()

    def get_tsne_(self, model, testloader):
        print('tsne start')
        model = model.eval()
        labels =[]
        imgs = None
        features = None


        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]
                model.module.mode = 'encoder'
                feat = model(data)
                labels.extend(test_label.cpu().numpy())

                current_imgs = data.cpu().numpy()
                if imgs is not None:
                    imgs = np.concatenate((imgs, current_imgs))
                else:
                    imgs = current_imgs

                current_features = feat.cpu().numpy()
                if features is not None:
                    features = np.concatenate((features, current_features))
                else:
                    features = current_features

        writer = tbx.SummaryWriter('runs/basedata')
        writer.add_embedding(features, metadata=labels, label_img=imgs)
        writer.close()
        print('tsne done')
        input()

    def get_gcam(self):
        model = self.model.module
        target_layers = [model.encoder.layer4[-1]]
        cam = GradCAM(model=model.encoder, target_layers=target_layers, use_cuda=True)
        paths=[]

        for i in range(2,self.args.sessions+1):
            lines = [x.strip() for x in open('./data/index_list/auo/session_{}.txt'.format(i), 'r').readlines()]
        for l in lines:
            paths.append(l)

        rgb_images = []
        input_tensors = []
        img_names=[]

        for path in paths:
            print(path)
            img_names.append(path.split('/')[1].split('_')[0])
            rgb_img = cv2.imread('/vol/AUO_Datasets/IL/data_1116/'+path, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (224, 224))
            rgb_img = np.float32(rgb_img) / 255
            rgb_images.append(rgb_img.copy())
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            input_tensors.append(input_tensor)

        input_tensor = torch.cat(input_tensors)

        target_category = None
        cam.batch_size = 128

        grayscale_cam = cam(input_tensor=input_tensor)

        for index, (cam, rgb_img, img_name) in enumerate(zip(grayscale_cam, rgb_images, img_names)):
            cam_image = show_cam_on_image(rgb_img, cam)
            cv2.imwrite('./grad_cam/' + f"{img_name}_{index}.png", cam_image)
            cv2.imwrite('./grad_cam/' + f"{img_name}_{index}_o.png", rgb_img*255)

    def set_save_path(self):
        pretrain_name = self.args.model_dir.split('/')[1].split('.')[0]
        # pretrain_name = self.args.model_dir.split('/')[5].split('.')[0]
        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        self.args.save_path = self.args.save_path + 'new_exp/'
        self.args.save_path = self.args.save_path + '%dW-%dS-%dQ-L%dW-L%dS-' % (
            self.args.episode_way, self.args.episode_shot, self.args.episode_query,
            self.args.low_way, self.args.low_shot)
        # if self.args.use_euclidean:
        #     self.args.save_path = self.args.save_path + '_L2/'
        # else:
        #     self.args.save_path = self.args.save_path + '_cos/'
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-%ds-%s' % (
                self.args.epochs_base, self.args.shot, pretrain_name)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-%ds-%s' % (
                self.args.epochs_base, self.args.shot, pretrain_name)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Epo_%d-Cosine-%ds-%s' % (
                self.args.epochs_base, self.args.shot, pretrain_name)
        
        if self.args.extra_class:
            self.args.save_path = self.args.save_path + '-extra'

        self.args.save_path = self.args.save_path + f'-ResEpoch_{self.args.epochs_res}'

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
