import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from PIL import Image
import glob
import imgaug.augmenters as iaa
import torchvision.transforms as transforms
from perlin import rand_perlin_2d_np

class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir+"/*/*.png"))
        self.resize_shape=resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0],image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}

        return sample



class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape

        self.image_paths = sorted(glob.glob(root_dir+"*.png"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"*.jpg"))

        # self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
        #               iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
        #               iaa.pillike.EnhanceSharpness(),
        #               iaa.AddToHueAndSaturation((-50,50),per_channel=True),
        #               iaa.Solarize(0.5, threshold=(32,128)),
        #               iaa.Posterize(),
        #               iaa.Invert(),
        #               iaa.pillike.Autocontrast(),
        #               iaa.pillike.Equalize(),
        #               iaa.Affine(rotate=(-45, 45))
        #               ]

        # self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.augmenters = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(45)], p=0.5),
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2))], p=0.5),
            transforms.RandomApply([transforms.RandomSolarize(threshold=128)], p=0.5),
            transforms.RandomApply([transforms.RandomPosterize(bits=4)], p=0.5),
            transforms.RandomApply([transforms.RandomInvert()], p=0.5),
            transforms.RandomApply([transforms.RandomAutocontrast()], p=0.5),
            transforms.RandomApply([transforms.RandomEqualize()], p=0.5)
        ])


    def __len__(self):
        return len(self.image_paths)


    def randAugmenter(self):
        # aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        # aug = iaa.Sequential([self.augmenters[aug_ind[0]],
        #                       self.augmenters[aug_ind[1]],
        #                       self.augmenters[aug_ind[2]]]
        #                      )
        # return aug
        # Returns a random augmenter from the list
        return transforms.Compose(np.random.choice(self.augmenters.transforms, 3, replace=False))

    def augment_image(self, image, anomaly_source_path):
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = Image.open(anomaly_source_path).convert("RGB")
        anomaly_source_img = anomaly_source_img.resize((self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = aug(anomaly_source_img)
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).item())
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).item())

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = np.expand_dims(perlin_noise, axis=2)
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))

        img_thr = np.array(anomaly_img_augmented).astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).item() * 0.8
        augmented_image = np.array(image).astype(np.float32) * (1 - perlin_thr) + (1 - beta) * img_thr + beta * np.array(image).astype(np.float32) * perlin_thr

        no_anomaly = torch.rand(1).item()
        if no_anomaly > 0.5:
            return np.array(image).astype(np.float32) / 255.0, np.zeros_like(perlin_thr, dtype=np.float32), torch.tensor([0.0], dtype=torch.float32)
        else:
            augmented_image = augmented_image.astype(np.float32) / 255.0
            msk = perlin_thr.astype(np.float32)
            has_anomaly = 1.0 if np.sum(msk) > 0 else 0.0
            return augmented_image, msk, torch.tensor([has_anomaly], dtype=torch.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.resize_shape[1], self.resize_shape[0]))

        if torch.rand(1).item() > 0.7:
            image = self.augmenters(image)

        image = np.array(image).astype(np.float32) / 255.0
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx])
        sample = {'image': image, "anomaly_mask": anomaly_mask, 'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
        return sample