# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 16:41:19 2019

@author: Jimmy Hua
"""

import numpy as np


class BaseDataset(object):
    """
    Base class of reid dataset
    """

    def get_imagedata_info(self, data):
        pids_l = list()
        
        for _, pid in data:
            pids_l.append(pid)
        #input()
        #print(pids_l)
        pids = set(pids_l)
        num_pids = len(pids)
        num_imgs = len(data)
        return num_pids, num_imgs

    def get_videodata_info(self, data, return_tracklet_stats=False):
        pids, tracklet_stats = [], [], []
        for img_paths, pid in data:
            pids += [pid]
            tracklet_stats += [len(img_paths)]
        pids = set(pids)
        num_pids = len(pids)
        num_tracklets = len(data)
        if return_tracklet_stats:
            return num_pids, num_tracklets, tracklet_stats
        return num_pids, num_tracklets

    def print_dataset_statistics(self):
        raise NotImplementedError


class BaseImageDataset(BaseDataset):
    """
    Base class of image reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ----------------------------------------")


class BaseVideoDataset(BaseDataset):
    """
    Base class of video reid dataset
    """

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_tracklets, num_train_cams, train_tracklet_stats = \
            self.get_videodata_info(train, return_tracklet_stats=True)

        num_query_pids, num_query_tracklets, num_query_cams, query_tracklet_stats = \
            self.get_videodata_info(query, return_tracklet_stats=True)

        num_gallery_pids, num_gallery_tracklets, num_gallery_cams, gallery_tracklet_stats = \
            self.get_videodata_info(gallery, return_tracklet_stats=True)

        tracklet_stats = train_tracklet_stats + query_tracklet_stats + gallery_tracklet_stats
        min_num = np.min(tracklet_stats)
        max_num = np.max(tracklet_stats)
        avg_num = np.mean(tracklet_stats)

        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # cameras")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:11d} | {:9d}".format(num_train_pids, num_train_tracklets, num_train_cams))
        print("  query    | {:5d} | {:11d} | {:9d}".format(num_query_pids, num_query_tracklets, num_query_cams))
        print("  gallery  | {:5d} | {:11d} | {:9d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_cams))
        print("  -------------------------------------------")
        print("  number of images per tracklet: {} ~ {}, average {:.2f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")
