# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class cuhkSYSU(ImageDataset):
    """CUHK SYSU datasets.

    The dataset is collected from two sources: street snap and movie.
    In street snap, 12,490 images and 6,057 query persons were collected
    with movable cameras across hundreds of scenes while 5,694 images and
    2,375 query persons were selected from movies and TV dramas.

    Dataset statistics:
        - identities: 11,934
        - images: 34,574
    """
    dataset_dir = 'cuhk_sysu'
    dataset_name = "cuhkSYSU"

    def __init__(self, root='datasets', **kwargs):
        self.root = '/data/pengyi/datasets/reid_data/'
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.data_dir = osp.join(self.dataset_dir, "cropped_images")

        required_files = [self.data_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.data_dir)
        query = []
        gallery = []

        super(cuhkSYSU, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dirname):
        img_paths = glob.glob(osp.join(dirname, '*.jpg'))
        # num_imgs = len(img_paths)

        # get all identities:
        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = img_name.split('_')[0]
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # num_pids = len(pid_container)

        # extract data
        data = []
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = img_name.split('_')[0]
            label = self.dataset_name + "_" + str(pid2label[pid])
            camid = self.dataset_name + "_0"
            data.append((img_path, label, camid)) # dummy camera id

        return data