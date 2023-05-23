import os
import numpy as np
import torch
from PIL import Image
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# base_path = "C:/Users/pavba/PycharmProjects/projekt_5/LoveDA_Train_16/Rural/"
# img_dir = base_path + "images_png/"
# mask_dir = base_path + "masks_png/"
IMG_LEN = 512
input_shape = (1024, 1024)
output_shape = (512, 512)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None, img_len=512):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_len = img_len

        self.img_dir_list_sorted = sorted(os.listdir(self.img_dir), key=len)
        self.mask_dir_list_sorted = sorted(os.listdir(self.mask_dir), key=len)

        self.len = len(os.listdir(self.img_dir))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_dir_list_sorted[idx]
        mask_path = self.mask_dir + self.mask_dir_list_sorted[idx]
        image = read_image(img_path)
        # image = image / 255

        mask_read = read_image(mask_path)
        # mask = torch.from_numpy(np.zeros((7, self.img_len, self.img_len)))
        # for i in range(1, 8):
        #     mask[i - 1, :, :] = (mask_read == i)
        #
        # if np.array(mask).max() == 0:
        #     return self.__getitem__(np.random.randint(0, len(self)))

        # image = image.numpy()

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            mask_read = self.target_transform(mask_read)

        # print(np.array(mask).shape)
        # print(torch.argmax(mask, dim=0))
        # print("mask: ", mask)

        # print(np.array(mask_read).shape)
        # print("mask_read: ",mask_read)
        mask_read = torch.squeeze(mask_read)
        return image, mask_read.long()
        # return image, np.squeeze(mask_read)