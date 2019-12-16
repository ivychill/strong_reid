# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:42:50 2019

@author: Jimmy Hua
"""
import glob
import os.path as osp

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

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid

class Kesci(BaseImageDataset):

    dataset_dir = 'match_2'

    def __init__(self, root, verbose=True, **kwargs):
        super(Kesci, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.train_dir = osp.join(self.dataset_dir, 'aug_train_list.txt')
        self.softmax_train_dir = osp.join(self.dataset_dir, 'train_softmax_list.txt')
        self.triplet_train_dir = osp.join(self.dataset_dir, 'aug_train_triplet_query_list.txt')
        # self.train_dir = osp.join(self.dataset_dir, 'plus_query.txt')
        self.query_dir = osp.join(self.dataset_dir, 'query_a_list.txt')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery_a')

        # self.query_a_dir = osp.join(self.dataset_dir, 'query_a_list.txt')
        # self.gallery_a_dir = osp.join(self.dataset_dir, 'gallery_a')

        self._check_before_run()

        # train = self._process_train(self.train_dir, relabel=True)
        softmax_train = self._process_train(self.softmax_train_dir, relabel=True)
        triplet_train = self._process_train(self.triplet_train_dir, relabel=True)
        query = self._process_train(self.query_dir, relabel=True)
        gallery = self._process_test(self.gallery_dir)
        # query_a = self._process_train(self.query_a_dir, relabel=True)
        # gallery_a = self._process_test(self.gallery_a_dir)

        if verbose:
            # self.print_dataset_statistics(train, query, gallery)
            self.print_dataset_statistics(softmax_train, query, gallery, triplet_train)

        # self.train = train
        self.softmax_train = softmax_train
        self.triplet_train = triplet_train
        self.query = query
        self.gallery = gallery

        # self.query_a = query_a
        # self.gallery_a = gallery_a

        # self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.train)
        self.num_train_pids, self.num_train_imgs = self.get_imagedata_info(self.softmax_train)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        # if not osp.exists(self.train_dir):
        #     raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.softmax_train_dir):
            raise RuntimeError("'{}' is not available".format(self.softmax_train_dir))
        if not osp.exists(self.triplet_train_dir):
            raise RuntimeError("'{}' is not available".format(self.triplet_train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

        # if not osp.exists(self.query_a_dir):
        #     raise RuntimeError("'{}' is not available".format(self.query_a_dir))
        # if not osp.exists(self.gallery_a_dir):
        #     raise RuntimeError("'{}' is not available".format(self.gallery_a_dir))
            
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

class Data():
    def __init__(self):
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

        dataset = Kesci(root=opt.data_path)
        self.num_classes = dataset.num_train_pids
        # self.train_set = ImageDataset(dataset.train, train_transform)
        self.softmax_train_set = ImageDataset(dataset.softmax_train, train_transform)
        self.triplet_train_set = ImageDataset(dataset.triplet_train, train_transform)
        self.query_set = ImageDataset(dataset.query, test_transform)
        # self.query_a_set = ImageDataset(dataset.query_a, test_transform)
        self.test_set = ImageDataset(dataset.query + dataset.gallery, test_transform)
        # self.test_a_set = ImageDataset(dataset.query_a + dataset.gallery_a, test_transform)
        self.query_paths = [q[0] for q in dataset.query]
        # self.query_a_paths = [ q[0] for q in dataset.query_a]
        self.gallery_paths = [g[0] for g in dataset.gallery]

        # for select test
        dataset.test = dataset.query + dataset.gallery
        self.test_paths = [t[0] for t in dataset.test]

        # self.triplet_train_loader = DataLoader(
        #     self.triplet_train_set, batch_size=opt.batch,
        #     sampler=RandomIdentitySampler(dataset.train, opt.batch, opt.instance),
        #     #sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
        #     num_workers=opt.num_workers)
        # self.softmax_train_loader = DataLoader(
        #     self.softmax_train_set, batch_size=opt.batch, shuffle=True, num_workers=opt.num_workers,)
        self.softmax_train_loader = DataLoader(
            self.softmax_train_set, batch_size=opt.batch,
            sampler=RandomIdentitySampler(dataset.softmax_train, opt.batch, opt.instance),
            num_workers=opt.num_workers,)
        self.triplet_train_loader = DataLoader(
            self.triplet_train_set, batch_size=opt.batch,
            sampler=RandomIdentitySampler(dataset.triplet_train, opt.batch, opt.instance),
            num_workers=opt.num_workers)
        self.query_loader = DataLoader(self.query_set, batch_size=opt.batch, shuffle=False, num_workers=opt.num_workers,)
        # self.query_a_loader = DataLoader(self.query_a_set, batch_size=opt.batch, shuffle=True, num_workers=opt.num_workers,)
        self.test_loader = DataLoader(self.test_set, batch_size=opt.batch*16, shuffle=False, num_workers=opt.num_workers,)
        # self.test_a_loader = DataLoader(self.test_a_set, batch_size=opt.batch*16, shuffle=False, num_workers=opt.num_workers,)
        # print('train:', len(self.train_set))
        # print('train:', len(self.train_loader))
        print('softmax_train:', len(self.softmax_train_set))
        print('softmax_train:', len(self.softmax_train_loader))
        print('triplet_train:', len(self.triplet_train_set))
        print('triplet_train:', len(self.triplet_train_loader))
        print('query:', len(self.query_set))
        print('query:', len(self.query_loader))
        print('test:', len(self.test_set))
        print('test:', len(self.test_loader))

# if __name__ == '__main__':
#
#     data = Data()