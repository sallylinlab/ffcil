import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class AUO(Dataset):

    def __init__(self, root='./data', train=True,
                 transform=None,
                 index_path=None, index=None, base_sess=None, mode=None, valid=False, extra=False, proto=False):
        self.IMAGE_PATH = '/mnt/backups/sun/lechien/auo_data'  
        # self.IMAGE_PATH = '/vol/AUO_Data_0414_DA/images'
        if train:
            setname = 'train'
        elif valid:
            setname = 'valid'
        elif extra:
            setname = 'extra'
            # self.IMAGE_PATH = '/home/lechien/A'
            # self.IMAGE_PATH = '/vol/AUO_Datasets/sp-1.3/IL/data_1116/B'
            self.IMAGE_PATH = '/hcds_vol/AUO_Datasets/sp-1.3/Data_released_20221205/01_fewshot/data/C120'
        elif proto:
            setname = 'proto'
        else:
            setname = 'test'
        
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.train = train  # training set or test set
        # self.IMAGE_PATH = '/home/lechien/auo_data'
        self.SPLIT_PATH = os.path.join(root, 'auo')

        csv_path = osp.join(self.SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb
        

        image_size = 224
        if train:
  
            self.transform = transforms.Compose([
                # transforms.Resize((image_size, image_size)),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            if base_sess:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)
            else:
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        elif extra:
            self.transform = transforms.Compose([
                # transforms.Resize((image_size, image_size)),
                # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)

        else:

            self.transform = transforms.Compose([
                # 256,256
                transforms.Resize([256, 256]),
                transforms.CenterCrop(image_size),
                # transforms.RandomResizedCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, index)


    def SelectfromTxt(self, data2label, index_path):
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        data_tmp = []
        targets_tmp = []
        for i in lines:
            img_path = os.path.join(self.IMAGE_PATH, i)
            try:
                targets_tmp.append(data2label[img_path])
                data_tmp.append(img_path)
            except Exception as e:
                print('wrong load txt')
                print(e) 
        
        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp = []
        targets_tmp = []
        for i in index:
            ind_cl = np.where(i == targets)[0]
            for j in ind_cl:
                try:
                    data_tmp.append(data[j])
                    targets_tmp.append(targets[j])
                except Exception as e:
                    print('wrong load classes')
                    print(e) 

        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        path, targets = self.data[i], self.targets[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, targets


if __name__ == '__main__':
    txt_path = "../../data/index_list/auo/session_1.txt"
    # class_index = open(txt_path).read().splitlines()
    base_class = 100
    class_index = np.arange(base_class)
    dataroot = '~/data'
    batch_size_base = 400
    trainset = AUO(root=dataroot, train=True, transform=None, index_path=txt_path)
    cls = np.unique(trainset.targets)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
                                              pin_memory=True)
    print(trainloader.dataset.data.shape)
    # txt_path = "../../data/index_list/cifar100/session_2.txt"
    # # class_index = open(txt_path).read().splitlines()
    # class_index = np.arange(base_class)
    # trainset = CIFAR100(root=dataroot, train=True, download=True, transform=None, index=class_index,
    #                     base_sess=True)
    # cls = np.unique(trainset.targets)
    # trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_base, shuffle=True, num_workers=8,
    #                                           pin_memory=True)
