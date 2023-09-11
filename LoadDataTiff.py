import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
from transform import ReLabel, ToLabel, Scale, HorizontalFlip, VerticalFlip, ColorJitter
import random
from scipy import ndimage

def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, root):
        self.PKS_size = (416,300)
        self.RLJ_size = (256,256)
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))
        self.img_resize_RLJ = Compose([
            Scale(self.RLJ_size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.img_resize_PKS = Compose([
            Scale(self.PKS_size, Image.BILINEAR),
            # We can do some colorjitter augmentation here
            # ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        ])
        self.label_resize_RLJ = Compose([
            Scale(self.RLJ_size, Image.NEAREST),
        ])
        self.label_resize_PKS = Compose([
            Scale(self.PKS_size, Image.NEAREST),
        ])
        self.img_transform = Compose([
            ToTensor(),
            Normalize(mean=[0.5],std=[0.225]),
        ])
        self.hsv_transform = Compose([
            ToTensor(),
        ])
        self.label_transform = Compose([
            ToLabel(),
            ReLabel(255, 1),
        ])
        self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.tif'.format("./data/train/dic"))))
        self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.tif'.format("./data/train/labels"))))
        self.name = os.path.basename(root)

        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No images/labels are found in {}".format(self.root))

    def __getitem__(self, index):
        use_sobel = False
        image = Image.open(self.input_paths[index]).convert('P')
        label = Image.open(self.label_paths[index]).convert('P')

        if 'PKS' in self.input_paths[index]:
            image = self.img_resize_PKS(image)
            label = self.label_resize_PKS(label)
        else:
            image = self.img_resize_RLJ(image)
            label = self.label_resize_RLJ(label)

        #randomly flip images
        if random.random() > 0.5:
            image = HorizontalFlip()(image)
            # image_hsv = HorizontalFlip()(image_hsv)
            label = HorizontalFlip()(label)
        if random.random() > 0.5:
            image = VerticalFlip()(image)
            # image_hsv = VerticalFlip()(image_hsv)
            label = VerticalFlip()(label)

        #randomly crop image to size 256*256
        w, h = image.size
        th, tw = (256,256)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if w == tw and h == th:
            image = image
            # image_hsv = image_hsv
            label = label
        else:
            if random.random() > 0.5:
                image = image.resize((256,256),Image.BILINEAR)
                # image_hsv = image_hsv.resize((128,128),Image.BILINEAR)
                label = label.resize((256,256),Image.NEAREST)
            else:
                image = image.crop((x1, y1, x1 + tw, y1 + th))
                # image_hsv = image_hsv.crop((x1, y1, x1 + tw, y1 + th))
                label = label.crop((x1, y1, x1 + tw, y1 + th))

        if use_sobel:
            norm_dic = np.array(image) - np.min(np.array(image))
            norm_dic = norm_dic /np.max(norm_dic)
            norm_dic*=255.0
            sobel_h = ndimage.sobel(norm_dic, 0)  # horizontal gradient
            sobel_v = ndimage.sobel(norm_dic, 1)  # vertical gradient
            sobel = np.sqrt(sobel_h**2 + sobel_v**2)
            sobel *= 255.0 / np.max(sobel)  # normalization
            sobel = Image.fromarray(sobel).convert('P')
            sobel = self.img_transform(sobel)   

        image = self.img_transform(image)
        if use_sobel:
            image = torch.cat([image,sobel],0)

        label = self.label_transform(label)

        return image, label

    def __len__(self):
        return len(self.input_paths)


class Dataset_val(torch.utils.data.Dataset):
    def __init__(self, root):
        size = (256,256) # resize input to fit into network
        self.root = root
        print(self.root)
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))
        self.img_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=[0.5],std=[0.225]),

        ])
        self.hsv_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
        ])
        self.label_transform = Compose([
            Scale(size, Image.NEAREST),
            ToLabel(),
            ReLabel(255, 1),
        ])
        self.input_paths = sorted(glob(os.path.join(self.root, '{}/*.tif'.format("./data/val/dic"))))
        self.label_paths = sorted(glob(os.path.join(self.root, '{}/*.tif'.format("./data/val/labels"))))

        self.name = os.path.basename(root)
        if len(self.input_paths) == 0 or len(self.label_paths) == 0:
            raise Exception("No images/labels are found in {}".format(self.root))

    def __getitem__(self, index):
        use_sobel = False
        image = Image.open(self.input_paths[index]).convert('P')
        # image_hsv = Image.open(self.input_paths[index]).convert('HSV')
        if use_sobel:
            norm_dic = np.array(image) - np.min(np.array(image))
            norm_dic = norm_dic /np.max(norm_dic)
            norm_dic*=255.0
            sobel_h = ndimage.sobel(norm_dic, 0)  # horizontal gradient
            sobel_v = ndimage.sobel(norm_dic, 1)  # vertical gradient
            sobel = np.sqrt(sobel_h**2 + sobel_v**2)
            sobel *= 255.0 / np.max(sobel)  # normalization
            sobel = Image.fromarray(sobel).convert('P')

        label = Image.open(self.label_paths[index]).convert('P')

        if self.img_transform is not None:
            image = self.img_transform(image)
            if use_sobel:
                sobel = self.img_transform(sobel)        
            # image_hsv = self.hsv_transform(image_hsv)
        else:
            image = image
            # image_hsv = image_hsv

        if self.label_transform is not None:
            label = self.label_transform(label)
        else:
            label = label
        if use_sobel:
            image = torch.cat([image,sobel],0)

        return image, label, self.input_paths[index]

    def __len__(self):
        return len(self.input_paths)

class Dataset_infer(torch.utils.data.Dataset):
    def __init__(self, root):
        size = (256,256)
        self.root = root
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(root))
        self.img_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=[0.5],std=[0.225]),

        ])
        self.hsv_transform = Compose([
            Scale(size, Image.BILINEAR),
            ToTensor(),
        ])
        self.label_transform = Compose([
            Scale(size, Image.NEAREST),
            ToLabel(),
            ReLabel(255, 1),
        ])
        #sort file names
        self.input_paths = sorted(glob(os.path.join(self.root, '*.tif')))
        self.name = os.path.basename(root)
        if len(self.input_paths) == 0:
            raise Exception("No images/labels are found in {}".format(self.root))

    def __getitem__(self, index):
        image = Image.open(self.input_paths[index]).convert('P')
        
        use_sobel = False
        if use_sobel:
            norm_dic = np.array(image) - np.min(np.array(image))
            norm_dic = norm_dic /np.max(norm_dic)
            norm_dic*=255.0
            sobel_h = ndimage.sobel(norm_dic, 0)  # horizontal gradient
            sobel_v = ndimage.sobel(norm_dic, 1)  # vertical gradient
            sobel = np.sqrt(sobel_h**2 + sobel_v**2)
            sobel *= 255.0 / np.max(sobel)  # normalization
            sobel = Image.fromarray(sobel).convert('P')

        if self.img_transform is not None:
            image = self.img_transform(image)
        else:
            image = image

        return image, self.input_paths[index] 

    def __len__(self):
        return len(self.input_paths)

def loader(dataset, batch_size, num_workers=0, shuffle=True):

    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)

    return input_loader
