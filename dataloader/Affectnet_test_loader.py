import numpy as np
import scipy.misc
import os
import pandas as pd
import torch
import cv2

class Affectnet_test_aligned(object):
    def __init__(self):

        src_labels = '/home/nlab/Desktop/'
        src_images = '/media/nlab/data/AffectNet_Aligned/' 
        training_csv = pd.read_csv(src_labels+'validation.csv')
        image_list = []
        label_list = []
        for index, row in training_csv.iterrows():
            if row['expression'] < 7 and row['subDirectory_filePath'].split('/')[1] in os.listdir(src_images+row['subDirectory_filePath'].split('/')[0]):
                image_list.append(src_images+row['subDirectory_filePath'])
                label_list.append(int(row['expression']))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        target = self.label_list[index]
        img = cv2.imread(img_path)

        # if len(img.shape) == 2:
        #     img = np.stack([img] * 3, 2)
        # flip = np.random.choice(2)*2-1
        # img = img[:, ::flip, :]
        # img = (img - 127.5) / 128.0
        # img = img.transpose(2, 0, 1)
        img = np.moveaxis(img, -1, 0) / 255.
        img = torch.from_numpy(img).float()

        return img, target

    # def get_image(self, img_data):
    #     path = img_data[0]

    #     img = cv2.imread(path)
    #     if img is None:
    #         print('no img found:', img_data)
    #         return None
    #     img = img[:, :, ::-1]
    #     # img = img[img_data[1]//2:(img_data[1]+img_data[3]+img.shape[0])//2,
    #     #           img_data[2]//2:(img_data[2]+img_data[4]+img.shape[1])//2]
    #     # img = cv2.resize(img, self.img_size)
    #     if self.train:
    #         img = self.aug(img)
    #     img = np.moveaxis(img, -1, 0) / 255.
    #     # print(np.all(np.isfinite(img)))
    #     return img

    def __len__(self):
        return len(self.image_list)



if __name__ == '__main__':
    dataset = Affectnet_aligned()
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)