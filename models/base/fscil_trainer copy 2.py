from os import sched_get_priority_max
from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *

from models.loss_function import *

from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score, precision_score, f1_score, recall_score

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        # torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
        # self.model = nn.parallel.DistributedDataParallel(self.model.cuda())
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()


        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            if 'dcl' in self.args.model_dir and 'swin' not in self.args.model_dir:
                self.best_model_dict = torch.load(self.args.model_dir)

                model_dict = {}
                print('--------------------------------------------------------------------------------------------')
                print('--------------------------------------------------------------------------------------------')



                for k,v in self.best_model_dict.items():

                    if 'module.model.0' in k:
                        model_dict[f'module.encoder.conv1{k[14]}{k[15:]}'] = v
                    elif 'module.model.1' in k:
                        model_dict[f'module.encoder.bn{k[13]}{k[14:]}'] = v
                    elif 'module.model' in k:
                        model_dict[f'module.encoder.layer{int(k[13])-3}{k[14:]}'] = v
                    # elif 'module.classifier_swap.weight' in k:
                    #     model_dict['module.fc.weight'] = v
                    else:
                        model_dict[k] = v
                self.best_model_dict = model_dict
                # self.best_model_dict = {f'module.encoder.{k[7:]}':v for k,v in self.best_model_dict.items()}

            elif 'swin' in self.args.model_dir:
                self.best_model_dict = torch.load(self.args.model_dir)
                model_dict = {}

                self.best_model_dict = model_dict

            elif 'resnet50' in self.args.model_dir:
                
                self.best_model_dict = torch.load(self.args.model_dir)
                # print(self.best_model_dict)
                # self.best_model_dict = {f'module.encoder{k[7:]}':v for k,v in self.best_model_dict}
                # self.best_model_dict = {k[11:]:v for k,v in self.best_model_dict.items()}
            else:
                self.best_model_dict = torch.load(self.args.model_dir)['params']
                
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())



    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.lr_base,
                                      momentum=0.9, nesterov=True, weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)
        
        self.criterion = None

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)
            # if self.args.dataset == 'auo':
            #     validset = args.Dataset.AUO(root=args.dataroot, train=False, index=np.arange(args.base_class), valid=True)
            #     validloader = torch.utils.data.DataLoader(
            #         dataset=validset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            # elif self.args.dataset == 'stanford_dogs':
            #     validset = args.Dataset.StanfordDogs(root=args.dataroot, train=False, index=np.arange(args.base_class), valid=True)
            #     validloader = torch.utils.data.DataLoader(
            #         dataset=validset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
            # else:
            #     validloader = testloader
            validloader = testloader
            msg = self.model.load_state_dict(self.best_model_dict, strict=False)
            print(msg)

            if session == 0:  # load base class train img label

                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()
                if self.args.opt_dir is not None:
                    print('Loading optimizer parameters from: %s' % self.args.opt_dir)
                    optimizer.load_state_dict(torch.load(self.args.opt_dir))
                s_t = time.strftime("%H%M", time.localtime())
                    
                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args, self.criterion)
                    # test model with all seen class
                    tsl, tsa, llist = test(self.model, validloader, epoch, args, session, self.criterion)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + f'_{s_t}.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        # torch.save(optimizer[0].state_dict(), os.path.join(args.save_path, f'optimizer_{s_t}.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    # scheduler[0].step()
                    # scheduler[1].step()
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    # self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    # best_model_dir = os.path.join(args.save_path, 'session' + str(session) + f'_{s_t}.pth')
                    # print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    # self.best_model_dict = deepcopy(self.model.state_dict())
                    # torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa, llist = test(self.model, testloader, 0, args, session, self.criterion)

                    recall = recall_score(llist['true_list'], llist['pred_list'], average=None)
                    result_list.append(f'all avg: {get_lavg(recall)}, base acc:{get_lavg(recall[:15])}, new acc:{get_lavg(recall[15:])}')
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                continue
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                tsl, tsa, llist = test(self.model, testloader, 0, args, session, self.criterion)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                # save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                # torch.save(dict(params=self.model.state_dict()), save_model_dir)
                # self.best_model_dict = deepcopy(self.model.state_dict())
                # print('Saving model to :%s' % save_model_dir)
                # print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, f'results_{s_t}.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def set_save_path(self):
        
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Cosine-Mix_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.mix)


        # if 'cos' in mode:
        #     self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)
        if self.args.model_dir != None:
            pretrain_name = self.args.model_dir.split('/')[1].split('.')[0]
            self.args.save_path = self.args.save_path + f'_{pretrain_name}'
        
        if self.args.extra_class:
            self.args.save_path = self.args.save_path + '-extra'

        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)
        if  self.args.model_dir != None:
            self.args.save_path = self.args.save_path + self.args.model_dir.split('/')[-1].split('.')[0]

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None
