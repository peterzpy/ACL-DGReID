# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""

import os
import os.path as osp
from glob import glob
from scipy.io import loadmat

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset
from fastreid.data.data_utils import read_json, write_json

__all__ = ['GRID', ]


# @DATASET_REGISTRY.register()
# class GRID(ImageDataset):
#     """GRID
#     """
#     dataset_dir = "underground_reid"
#     dataset_name = 'grid'

#     def __init__(self, root='datasets', **kwargs):
#         self.root = root
#         self.train_path = os.path.join(self.root, self.dataset_dir, 'images')

#         required_files = [self.train_path]
#         self.check_before_run(required_files)

#         train = self.process_train(self.train_path)

#         super().__init__(train, [], [], **kwargs)

#     def process_train(self, train_path):
#         data = []
#         img_paths = glob(os.path.join(train_path, "*.jpeg"))

#         for img_path in img_paths:
#             img_name = os.path.basename(img_path)
#             img_info = img_name.split('_')
#             pid = self.dataset_name + "_" + img_info[0]
#             camid = self.dataset_name + "_" + img_info[1]
#             data.append([img_path, pid, camid])
#         return data


@DATASET_REGISTRY.register()
class GRID(ImageDataset):
    """GRID.
    Reference:
        Loy et al. Multi-camera activity correlation analysis. CVPR 2009.
    URL: `<http://personal.ie.cuhk.edu.hk/~ccloy/downloads_qmul_underground_reid.html>`_
    
    Dataset statistics:
        - identities: 250.
        - images: 1275.
        - cameras: 8.
    """
    dataset_dir = 'underground_reid'
    dataset_name = 'grid'
    _junk_pids = [0]

    def __init__(self, root='', split_id=0, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.probe_path = osp.join(
            self.dataset_dir, 'probe'
        )
        self.gallery_path = osp.join(
            self.dataset_dir, 'gallery'
        )
        self.split_mat_path = osp.join(
            self.dataset_dir, 'features_and_partitions.mat'
        )
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        required_files = [
            self.dataset_dir, self.probe_path, self.gallery_path,
            self.split_mat_path
        ]
        self.check_before_run(required_files)

        self.prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                'split_id exceeds range, received {}, '
                'but expected between 0 and {}'.format(
                    split_id,
                    len(splits) - 1
                )
            )
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        super(GRID, self).__init__(train, query, gallery, **kwargs)

    def prepare_split(self):
        if not osp.exists(self.split_path):
            print('Creating 10 random splits')
            split_mat = loadmat(self.split_mat_path)
            trainIdxAll = split_mat['trainIdxAll'][0] # length = 10
            probe_img_paths = sorted(
                glob.glob(osp.join(self.probe_path, '*.jpeg'))
            )
            gallery_img_paths = sorted(
                glob.glob(osp.join(self.gallery_path, '*.jpeg'))
            )

            splits = []
            for split_idx in range(10):
                train_idxs = trainIdxAll[split_idx][0][0][2][0].tolist()
                assert len(train_idxs) == 125
                idx2label = {
                    idx: label
                    for label, idx in enumerate(train_idxs)
                }

                train, query, gallery = [], [], []

                # processing probe folder
                for img_path in probe_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    img_idx = self.dataset_name + "_" + str(img_idx)
                    camid = self.dataset_name + "_" + str(camid)
                    if img_idx in train_idxs:
                        train.append([img_path, idx2label[img_idx], camid])
                    else:
                        query.append([img_path, img_idx, camid])

                # process gallery folder
                for img_path in gallery_img_paths:
                    img_name = osp.basename(img_path)
                    img_idx = int(img_name.split('_')[0])
                    camid = int(
                        img_name.split('_')[1]
                    ) - 1 # index starts from 0
                    img_idx = self.dataset_name + "_" + str(img_idx)
                    camid = self.dataset_name + "_" + str(camid)
                    if img_idx in train_idxs:
                        train.append([img_path, idx2label[img_idx], camid])
                    else:
                        gallery.append([img_path, img_idx, camid])

                split = {
                    'train': train,
                    'query': query,
                    'gallery': gallery,
                    'num_train_pids': 125,
                    'num_query_pids': 125,
                    'num_gallery_pids': 900
                }
                splits.append(split)

            print('Totally {} splits are created'.format(len(splits)))
            write_json(splits, self.split_path)
            print('Split file saved to {}'.format(self.split_path))