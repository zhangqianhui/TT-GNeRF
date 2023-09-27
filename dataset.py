import os
import csv
import lmdb
import random
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
import glob
import torchvision.transforms as transforms
import PIL

class FFHQFake(Dataset):
    """CelelebA Dataset"""

    def __init__(self, dataset_path, transform, resolution, csvfile='', **kwargs):
        super().__init__()

        print(csvfile)

        self.dataset_path = dataset_path
        # self.data = glob.glob(dataset_path)
        #
        # assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"

        self.transform = transform
        self.resolution = resolution

        with open(csvfile, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        self.label_list = []
        self.style_list = []

        #ratio

        # hair color
        # label1 32019
        # label2 7981

        # gender
        # label1 9254  # 3.32
        # label2 30746

        # bangs
        # label1 37992 # 0.05
        # label2 2008

        # Age
        # label1 13476 # 1.96
        # label2 26524

        # smile
        # label1 15718 # 1.54
        # label2 24282

        # bears:
        # label1 13915 # 1.87
        # label2 26085

        for j in range(len(data)):
            item_list = []
            items = data[j][0].split(' ')
            for i in range(len(items)):
                if i not in [0]:
                    if float(items[i]) > 0.8:
                        item_list.append(1)
                    else:
                        item_list.append(0)

            self.label_list.append(item_list)
            self.style_list.append(items[0])

        print(len(self.label_list))

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):

        imgname = self.style_list[index]

        print(imgname)

        datapath = self.dataset_path.split('*.png')[0]

        print(datapath)

        imgpath = os.path.join(datapath, imgname)

        print(imgpath)

        styles_path = imgpath.split('.png')[0] + '_ws.npy'

        print(styles_path)

        labels = np.array(self.label_list[index])

        styles = np.load(styles_path).squeeze()
        X = PIL.Image.open(imgpath)
        img = self.transform(X)

        # pose
        pose_path = imgpath.split('.png')[0] + '_pose.npy'
        poses = np.load(pose_path).squeeze()

        return img, labels, styles, poses

