# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:42:50 2019

@author: Jimmy Hua
"""
import glob
import os.path as osp
import os

import torchvision.transforms as T
from .bases import BaseImageDataset
from .transforms import RandomErasing,Random2DTranslation
from .sampler import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from opt import opt
from PIL import Image
from torch.utils.data import Dataset, DataLoader


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None,valid=False):
        self.dataset = dataset
        self.transform = transform
        self.valid = valid

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.valid:
            img_path, pid = self.dataset[index]
        else:
            img_path, pid ,camid= self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        if not self.valid:
            return img, pid
        else:
            return img,pid,camid

class Kesci(BaseImageDataset):

    dataset_dir = 'sz_reid'

    def __init__(self, root='/home/kcadmin/user/fengchen/reid/dataset', valid=False,verbose=True, **kwargs):
        super(Kesci, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train.txt')
        print(self.train_dir )
        self.query_dir = osp.join(self.dataset_dir, 'query_a_list.txt')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_a')
        if valid:
            self.val_query_dir = osp.join(self.dataset_dir,'query.txt')
            self.val_gallery_dir = osp.join(self.dataset_dir,'gallery.txt')

        self._check_before_run()

        train = self._process_train(self.train_dir, relabel=True)
        query = self._process_train(self.query_dir, relabel=True)
        gallery = self._process_test(self.gallery_dir)
        if valid:
            val_query = self._process_valid(self.val_query_dir,camid=1)
            val_gallary = self._process_valid(self.val_gallery_dir,camid=2)

        if verbose:
            print("=> Kesci loaded")
            self.print_dataset_statistics(train, query, gallery)
            if valid:
                self.print_dataset_statistics(train, val_query, val_gallary)

        self.train = train
        self.query = query
        self.gallery = gallery
        if valid:
            self.val_query = val_query
            self.val_gallery = val_gallary

        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
            
    def _process_test(self, dir_path):
        dataset = []
        img_paths = glob.glob(osp.join(dir_path, '*.png'))

        for p in img_paths:
            dataset.append((p, 0))

        return dataset
        

    def _process_train(self, dir_path, relabel=False):
        dataset = []
        with open(dir_path, 'r') as f:
            lines = f.readlines()
        _,dataset_dir = os.path.split(dir_path)
        pid_container = set([int(p.strip().split()[-1]) for p in lines])
        #print(pid_container)
        for line in lines:
            img_name = line.strip().split()[0]
            img_path = osp.join(self.dataset_dir, img_name)

            pid = int(line.strip().split()[-1])

            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid))
        return dataset

    def _process_valid(self, dir_path, relabel=False,camid=0):
        dataset = []
        with open(dir_path, 'r') as f:
            lines = f.readlines()
        _,dataset_dir = os.path.split(dir_path)
        pid_container = set([int(p.strip().split()[-1]) for p in lines])
        #print(pid_container)
        for line in lines:
            img_name = line.strip().split()[0]
            img_path = osp.join(self.dataset_dir, img_name)

            pid = int(line.strip().split()[-1])

            pid2label = {pid: label for label, pid in enumerate(pid_container)}

            if relabel: pid = pid2label[pid]

            dataset.append((img_path, pid,camid))
        return dataset

class Data():
    def __init__(self,valid=opt.evaluate):
        train_transform = T.Compose([
            T.Resize((256, 128), interpolation=3),
            T.RandomHorizontalFlip(p=0.5),
            Random2DTranslation(height=256, width=128, p = 0.5),
            T.Pad(10),
            T.RandomCrop([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])

        test_transform = T.Compose([
            T.Resize((256, 128), interpolation=3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        dataset = Kesci(root=opt.data_path,valid=valid)
        self.num_classes = dataset.num_train_pids
        self.train_set = ImageDataset(dataset.train, train_transform)
        self.test_set = ImageDataset(dataset.query + dataset.gallery, test_transform)
        self.query_paths = [ q[0] for q in dataset.query]
        self.gallery_paths = [ g[0] for g in dataset.gallery]
        if valid:
            self.val_set = ImageDataset(dataset.val_query + dataset.val_gallery, test_transform,valid=True)
            self.val_query_paths = [ q[0] for q in dataset.val_query]
            self.val_gallery_paths = [ q[0] for q in dataset.val_gallery]

        self.train_loader = DataLoader(
            self.train_set, batch_size=opt.batch,
            sampler=RandomIdentitySampler(dataset.train, opt.batch, opt.instance),
            #sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=opt.num_workers)
        self.test_loader = DataLoader(self.test_set, batch_size=opt.batch//2, shuffle=False, num_workers=opt.num_workers,)
        print('train:', len(self.train_set))
        print('train:', len(self.train_loader))
        print('test:', len(self.test_set))
        print('test:', len(self.test_loader))
        if valid:
            self.val_loader = DataLoader(self.val_set, batch_size=opt.batch//2, shuffle=False,
                                         num_workers=opt.num_workers)
            print('valid:', len(self.val_set))
            print('valid:', len(self.val_loader))

# if __name__ == '__main__':
#
#     data = Data()