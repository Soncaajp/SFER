import numpy as np
import scipy.misc
import os
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt



class Affectnet_aligned(object):
    def __init__(self, transform = None):

        src_labels = '/home/nlab/Desktop/'
        src_images = '/media/nlab/data/AffectNet_Aligned/' 
        training_csv = pd.read_csv(src_labels+'training.csv')
        image_list = []
        label_list = []
        for index, row in training_csv.iterrows():
            if row['expression'] < 7 and row['subDirectory_filePath'].split('/')[1] in os.listdir(src_images+row['subDirectory_filePath'].split('/')[0]):
                image_list.append(src_images+row['subDirectory_filePath'])
                label_list.append(int(row['expression']))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = cv2.imread(img_path)
        img = np.moveaxis(img, -1, 0)

        if self.transform:
            img = self.transform(img)
        img = img / 255.
        img = torch.from_numpy(img).float()
        return img, target

    def __len__(self):
        return len(self.image_list)



if __name__ == '__main__':
    dataset = Affectnet_aligned()
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
        print(data[1])
