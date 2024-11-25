from ntpath import join
import gc
import random
import heapq
import os.path as osp

import torch.nn as nn
import torchvision
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

class FSCILTrainer(Trainer):
    def __init__(self, args):

        super().__init__(args)

        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.set_up_model()


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
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=True,
                                                  index=class_index, base_sess=True)
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_index, base_sess=True)
        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'auo':
            trainset = self.args.Dataset.AUO(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.AUO(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'stanford_dogs':
            trainset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=False, index=class_index)           
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
        if self.args.dataset == 'stanford_dogs':
            trainset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=True,
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
        if self.args.dataset == 'stanford_dogs':
            testset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=False, index=class_new)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=self.args.num_workers, pin_memory=True)

        return trainset, trainloader, testloader


    def get_session_classes(self, session):
        class_list = np.arange(self.args.base_class + session * self.args.way)
        return class_list

    def replace_to_rotate(self, proto_tmp, query_tmp):

        transform = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(brightness=(0.5 , 1.2), contrast=(0.5,1.5), saturation=(0.5,1.5), hue=(-0.1, 0.1))
        ])


        # random choose rotate or color augmentation
        for i in range(self.args.low_way):
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


    def get_optimizer_res(self):
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
                print('new classes for this session:\n', np.unique(train_set.targets))

                # res training
                print('start res train')
                optimizer, scheduler = self.get_optimizer_res()
                result_list.append('\nres training log')

                for epoch in range(args.epochs_res):
                    start_time = time.time()
                    # train base sess
                    self.model.eval()
                    tl, ta = self.res_train(self.model, trainloader, optimizer, scheduler, epoch, args)

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

                    lrc = scheduler.get_last_lr()[0]
                    print('epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                        epoch, lrc, tl, ta, vl, va))
                    result_list.append(
                        'epoch:%03d,lr:%.5f,training_loss:%.5f,training_acc:%.5f,val_loss:%.5f,val_acc:%.5f' % (
                            epoch, lrc, tl, ta, vl, va))
                    print('still need around %.2f mins to finish' % (
                                  (time.time() - start_time) * (args.epochs_res - epoch) / 60))
                    scheduler.step()



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


    def validation(self):
        with torch.no_grad():
            model = self.model

            for session in range(1, self.args.sessions):
                train_set, trainloader, testloader = self.get_dataloader(session)
                # if self.args.dataset == 'auo':
                #     validset = self.args.Dataset.AUO(root=self.args.dataroot, train=False, index=self.get_session_classes(session), valid=True)
                #     validloader = torch.utils.data.DataLoader(dataset=validset, batch_size=128, shuffle=False,
                #                                             num_workers=self.args.num_workers, pin_memory=True)
                # elif self.args.dataset == 'stanford_dogs':
                #     validset = self.args.Dataset.StanfordDogs(root=self.args.dataroot, train=False, index=self.get_session_classes(session), valid=True)
                #     validloader = torch.utils.data.DataLoader(dataset=validset, batch_size=128, shuffle=False,
                #                                             num_workers=self.args.num_workers, pin_memory=True)
                # else:
                #     validloader = testloader
                validloader = testloader

                trainloader.dataset.transform = testloader.dataset.transform
                model.module.mode = 'avg_cos'
                model.eval()
                model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                vl, va = self.test(model, validloader, self.args, session)

        return vl, va


    def res_train(self, model, trainloader, optimizer, scheduler, epoch, args):
        tl = Averager()
        ta = Averager()
        tqdm_gen = tqdm(trainloader)

        label = torch.arange(start=args.base_class, end=args.base_class+args.low_way).repeat(args.episode_query)
        label = label.type(torch.cuda.LongTensor)

        triplet_loss = torch.nn.TripletMarginLoss(margin=1.0, p=2)
        
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
                    


            proto, query = data[:k], data[k:]
            proto_tmp, query_tmp = self.get_rotate_pseudo_classes(proto, query)
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

            anchor = torch.cat([proto, proto_tmp], dim=2).view(args.episode_way*2, proto.shape[-1])
            query_trip = query.view(args.episode_query*2, args.episode_way, query.shape[-1])
            positive, negative = self.get_triplet_loss_data(query_trip)
            t_loss = triplet_loss(anchor, positive, negative)

            total_loss = F.cross_entropy(logits, true_label) + 0.2*t_loss
            total_loss = F.mse_loss(logits, F.one_hot(true_label, num_classes = args.base_class+args.low_way).float())
            acc = count_acc(logits, true_label)

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


                proto = proto.unsqueeze(0).unsqueeze(0)
                logits = model.module._forward(proto, query)
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)

                p = torch.argmax(logits, dim=1).cpu().numpy()
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

        # get each class accuracy
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

                


        return vl, va



    def set_save_path(self):
        pretrain_name = self.args.model_dir.split('/')[-1].split('.')[0]
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
