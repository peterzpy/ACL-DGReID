# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.solver.optim import lamb
from torch.utils.data import Dataset

from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""
    #CHANGE Add domain id

    def __init__(self, img_items, transform=None, relabel=True, mapping=None, offset=0):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel
        self.mapping = mapping

        assert self.mapping is not None, 'mapping must be initialized!!!'

        if isinstance(self.mapping, dict):
            pid_set = [set() for i in range(len(self.mapping))]
            cam_set = [set() for i in range(len(self.mapping))]
            for i in img_items:
                domain_id = self.mapping[i[1].split("_")[0]]
                pid_set[domain_id].add(i[1])
                cam_set[domain_id].add(i[2])

            self.pids = [] 
            self.cams = [] 
            for temp_pid, temp_cam in zip(pid_set, cam_set):
                self.pids += sorted(list(temp_pid))
                self.cams += sorted(list(temp_cam))
        else:
            pid_set = set()
            cam_set = set()
            for i in img_items:
                pid_set.add(i[1])
                cam_set.add(i[2])

            self.pids = sorted(list(pid_set))
            self.cams = sorted(list(cam_set))
        
        if relabel:
            self.pid_dict = dict([(p, i+offset) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_item = self.img_items[index]
        img_path = img_item[0]
        pid = img_item[1]
        camid = img_item[2]
        img = read_image(img_path)
        if self.transform is not None:
            img0 = self.transform[0](img)
            img = self.transform[1](img)
        if self.mapping and isinstance(self.mapping, dict):
            domain_id = self.mapping[pid.split("_")[0]]
        else:
            domain_id = self.mapping
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images0": img0,
            "images": img,
            "targets": pid,
            "camids": camid,
            "domainids": domain_id,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)
