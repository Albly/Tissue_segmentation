import os
import sys
import numpy as np
import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import albumentations as A


class HistoDatasetRandom(Dataset):
    def __init__(self, folder, names, mode="val", augment=False, img_size=256, preload=False, mask_threshold=128):
        self.folder = folder
        self.names = names
        self.mode = mode
        self.augment = augment
        self.img_size = img_size
        self.mask_threshold = mask_threshold

        if self.mode == "train" and self.augment:
            self.transform = A.Compose([
                # A.RandomResizedCrop(img_size, img_size),
                # A.Rotate(),
                # A.Flip(),
                # A.ColorJitter(),
                A.RandomCrop(height=img_size, width=img_size, always_apply = True),
                A.HorizontalFlip(always_apply = False, p = 0.5),
                A.VerticalFlip(always_apply = False, p = 0.5),
                A.ColorJitter(),
            ])
        else:
            self.transform = A.RandomCrop(img_size, img_size)

        self.to_tensor = ToTensor()

        # if preload:
        self.preload = preload
        self.load_images()
        # else:
        #     self.images = None
        #     self.masks = None

    def __len__(self):
        if self.indexes_adjusted is not None:
            return len(self.indexes_adjusted)
        else:
            return len(self.images)

    def load_images(self):
        self.images = []
        self.masks = []
        self.image_sizes = []
        self.indexes_adjusted = []
        for index, name in enumerate(self.names):
            filename_image = os.path.join(self.folder, name + ".jpg")
            filename_mask = os.path.join(self.folder, name + "_mask.jpg")
            # image = np.asarray(Image.open(filename_image))
            # mask = np.asarray(Image.open(filename_mask))
            image = cv2.imread(filename_image).astype(np.float32) / 255
            # image = np.rollaxis(image, -1, 0)
            mask = cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE)
            if self.preload:
                self.images.append(image)
                self.masks.append(mask)
            self.image_sizes.append(image.shape[:2])
            self.indexes_adjusted += [index]*int(np.ceil(image.shape[0]*image.shape[1] / self.img_size**2))
        if not self.preload:
            self.images = None
            self.masks = None

    def __getitem__(self, index):
        if self.indexes_adjusted is not None:
            index = self.indexes_adjusted[index]
        if self.images is None:
            filename_image = os.path.join(self.folder, self.names[index] + ".jpg")
            filename_mask = os.path.join(self.folder, self.names[index] + "_mask.jpg")
            # image = np.asarray(Image.open(filename_image))
            # mask = np.asarray(Image.open(filename_mask))
            image = cv2.imread(filename_image).astype(np.float32) / 255
            # image = np.rollaxis(image, -1, 0)
            mask = cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE)
        else:
            image, mask = self.images[index], self.masks[index]

        # image = image.astype(np.float32) / 255

        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        image = self.to_tensor(image.copy())
        mask = torch.from_numpy(mask >= self.mask_threshold).float()

        return image, mask


class HistoDatasetAll(Dataset):
    def __init__(self, folder, names, img_size=256, mask_threshold=128):
        self.folder = folder
        self.names = names
        self.img_size = img_size
        self.mask_threshold = mask_threshold

        self.to_tensor = ToTensor()

        self.load_images()

    def load_images(self):
        self.images = []
        self.masks = []
        self.image_sizes = []
        self.indexes_adjusted = []
        self.indexes_images = []
        self.indexes_adjusted_2 = []
        for index, name in enumerate(self.names):
            filename_image = os.path.join(self.folder, name + ".jpg")
            filename_mask = os.path.join(self.folder, name + "_mask.jpg")
            # image = np.asarray(Image.open(filename_image))
            # mask = np.asarray(Image.open(filename_mask))
            image = cv2.imread(filename_image).astype(np.float32) / 255
            # image = np.rollaxis(image, -1, 0)
            mask = cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE)
            self.images.append(image)
            self.masks.append(mask)
            self.image_sizes.append(image.shape[:2])
            inds = []
            for i in range(int(np.ceil(image.shape[0] / self.img_size))):
                for j in range(int(np.ceil(image.shape[1] / self.img_size))):
                    ind_i = min(self.img_size * i, image.shape[0] - self.img_size)
                    ind_j = min(self.img_size * j, image.shape[1] - self.img_size)
                    inds.append([ind_i, ind_j])
            self.indexes_adjusted += [index]*len(inds)
            self.indexes_adjusted_2 += np.arange(len(inds)).tolist()
            self.indexes_images += inds

    def __len__(self):
        return len(self.indexes_adjusted) // 10

    def __getitem__(self, index):
        index_image = self.indexes_adjusted[index]
        index_sub = self.indexes_adjusted_2[index]
        indexes_origin = self.indexes_images[index]

        image, mask = self.images[index_image], self.masks[index_image]
        image = image[indexes_origin[0]:indexes_origin[0]+self.img_size, 
            indexes_origin[1]:indexes_origin[1]+self.img_size]
        mask = mask[indexes_origin[0]:indexes_origin[0]+self.img_size, 
            indexes_origin[1]:indexes_origin[1]+self.img_size]

        # image = self.to_tensor(image.astype(np.float32) / 255)
        image = self.to_tensor(image.astype(np.float32))
        mask = torch.from_numpy(mask >= self.mask_threshold).float()

        indexes_origin = np.array(indexes_origin, np.int32)
        
        return image, mask, index_image, indexes_origin


def get_dataloader(dataset, **kwargs):
    return DataLoader(dataset, **kwargs)


def recreate_full_image(images, indexes):
    size = [0, 0]
    for index in indexes:
        size[0] = max(size[0], index[0]+images[0].shape[0])
        size[1] = max(size[1], index[1]+images[0].shape[1])
    if len(images[0].shape) == 2:
        new_image = np.zeros(size, np.float32)
    else:
        new_image = np.zeros((size[0], size[1], 3), np.float32)
    for index, image in zip(indexes, images):
        if len(image.shape) == 2:
            new_image[index[0]:index[0]+image.shape[0], index[1]:index[1]+image.shape[1]] = \
                np.maximum(new_image[index[0]:index[0]+image.shape[0], index[1]:index[1]+image.shape[1]],
                    image)
        else:
            image = np.swapaxes(image, 0, 2)
            try:
                new_image[index[0]:index[0]+image.shape[0], index[1]:index[1]+image.shape[1]] = image
            except:
                pass
    return new_image

def recreate_full_image_list(images, indexes, indexes_images):
    images_new = []
    for index in np.unique(indexes):
        inds = np.where(indexes == index)[0]
        images_new.append(recreate_full_image(images[inds], indexes_images[inds]))
    return images_new