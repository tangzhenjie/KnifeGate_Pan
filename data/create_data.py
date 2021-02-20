import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import os
import random
import numpy as np


from PIL import Image as m

def transform(image, mask):

    # Random horizontal flipping
    if random.random() > 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = TF.vflip(image)
        mask = TF.vflip(mask)

    # resize
    image = image.resize((512, 256), resample=m.BICUBIC)
    mask = mask.resize((512, 256), resample=m.NEAREST)
    mask = np.array(mask).astype(np.long)

    # Transform to tensor
    nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    image = TF.to_tensor(image)
    image = nomal_fun_image(image)


    mask = TF.to_tensor(mask)

    return image, mask

class TrainDataset(data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir + "images"
        self.label_dir = image_dir + "labels"
        self.image_paths = sorted([os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir)])
        self.label_paths = sorted([os.path.join(self.label_dir, x) for x in os.listdir(self.label_dir)])

        self.length = len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        img = m.open(image_path).convert('RGB')
        label = m.open(label_path).convert('L')

        img, label = transform(img, label)

        return img, label
    def __len__(self):
        return self.length

class TestDataset(data.Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir + "images"
        self.image_paths = sorted([os.path.join(self.image_dir, x) for x in os.listdir(self.image_dir)])

        self.length = len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        img = m.open(image_path).convert('RGB')
        img = img.resize((512, 256), resample=m.BICUBIC)

        nomal_fun_image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        img = TF.to_tensor(img)
        img = nomal_fun_image(img)


        return img
    def __len__(self):
        return self.length