import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image, ImageFilter
import random
from torchvision.transforms import functional as F

import moco
from .randaugment import rand_augment_transform
import copy
import scipy.stats as st
import torch


class ImageNetLT_moco(Dataset):
    def __init__(self, root,mix_root, txt,num_classes, transform=None):
        self.img_path = []
        self.labels = []
        self.root=root
        self.mix_root=mix_root
        self.num_classes=num_classes
        # normalize = transforms.Normalize(mean=[0.765, 0.545, 0.569],
        #                                  std=[0.140, 0.151, 0.169])  #isic
        normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # normalize = transforms.Normalize(mean=[0.461, 0.247, 0.080],
        #                                  std=[0.250, 0.139, 0.080])   #aptos
        # normalize_gray = transforms.Normalize((0.017,), (0.126,))  #canny
        # normalize_gray = transforms.Normalize((0.574,), (0.363,))  #lbp
        # normalize_gray = transforms.Normalize((0.291,), (0.453,))  #mask
        normalize_gray = transforms.Normalize((0.144,), (0.137,))  #isic saliency
        # normalize_gray = transforms.Normalize((0.072,), (0.122,))  #aptos saliency

        self.resized_crop=transforms.RandomResizedCrop((224,224), scale=(0.08, 1.))
        self.resized_crop_random=transforms.RandomResizedCrop((224,224))

        self.origin_transform=transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize
        ])
        self.pre_transform=transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            normalize_gray
        ])
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        self.class_data=[[] for i in range(self.num_classes)]
        for i in range(len(self.labels)):
            y=self.labels[i]
            self.class_data[y].append(i)

        self.cls_num_list=[len(self.class_data[i]) for i in range(self.num_classes)]

    def __len__(self):
        return len(self.labels)
    """
        为了得到两张图(其中一张是随机选取的)的图像和索引值信息
    """

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        rgb_mean = (0.7893, 0.5683, 0.5930)
        ra_params = dict(translate_const=int(224 * 0.45),
                         img_mean=tuple([min(255, round(255 * x)) for x in rgb_mean]), )
        path = self.img_path[index]
        target = self.labels[index]
        mix_path = path.replace(self.root, self.mix_root)
        with open(path, 'rb') as f:
            img_origin = Image.open(f).convert('RGB')
        with open(mix_path, 'rb') as f:
            img_pre = Image.open(f).convert('RGB')


        img_origin1,img_pre1=copy.copy(img_origin),copy.copy(img_pre)
        origin_sample1,img_sample1=self.resized_crop_with_preKnowledge(self.resized_crop,img_origin1,img_pre1)
        origin_sample1, img_sample1=self.RandomHorizonalFlip(origin_sample1,img_sample1,p=0.5)
        origin_sample1=transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
            ], p=1.0)(origin_sample1)
        origin_sample1, img_sample1 = rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params,
                                                             origin_sample1, img_sample1)
        origin_sample1=self.origin_transform(origin_sample1)
        img_sample1=self.pre_transform(img_sample1)


        origin_sample2,img_sample2=self.resized_crop_random_with_preKnowledge(self.resized_crop_random,img_origin,img_pre)
        origin_sample2, img_sample2=self.RandomHorizonalFlip(origin_sample2,img_sample2,p=0.5)
        origin_sample2 = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.0)
        ], p=1.0)(origin_sample2)
        origin_sample2=transforms.RandomGrayscale(p=0.2)(origin_sample2)
        origin_sample2=transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5)(origin_sample2)
        origin_sample2, img_sample2 = rand_augment_transform('rand-n{}-m{}-mstd0.5'.format(2, 10), ra_params,
                                                             origin_sample2, img_sample2)
        origin_sample2=self.origin_transform(origin_sample2)
        img_sample2=self.pre_transform(img_sample2)
        return [origin_sample1, origin_sample2,img_sample1,img_sample2], target 


    def resized_crop_with_preKnowledge(self, tf, origin, img):
        assert isinstance(tf, transforms.RandomResizedCrop)
        i, j, h, w = tf.get_params(origin, tf.scale, tf.ratio)
        img = F.resized_crop(img, i, j, h, w, tf.size, tf.interpolation)
        origin = F.resized_crop(origin, i, j, h, w, tf.size, tf.interpolation)
        return origin, img

    def RandomHorizonalFlip(self, origin, img, p):
        if random.random() < p:
            img = F.hflip(img)
            origin = F.hflip(origin)
        return origin, img

    def resized_crop_random_with_preKnowledge(self, tf, origin, img):
        assert isinstance(tf, transforms.RandomResizedCrop)
        i, j, h, w = tf.get_params(origin, tf.scale, tf.ratio)
        img = F.resized_crop(img, i, j, h, w, tf.size, tf.interpolation)
        origin = F.resized_crop(origin, i, j, h, w, tf.size, tf.interpolation)
        return origin, img
