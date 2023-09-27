import numpy as np
import os
import logging
import torch

from .base import BaseSimulator, IntractableLikelihoodError
from .utils import NumpyDataset
from torch.utils.data import random_split, Dataset
import random
from tqdm import tqdm 

logger = logging.getLogger(__name__)

class BlobsManifoldSimulator(BaseSimulator):
    """ MNIST in vector format """

    def __init__(self, args):
        super().__init__()

        self._args = args
        self._image_size = args.image_size
        self._latent_dim = args.latent_dim
        self._split_ratio = args.split_ratio

    def latent_dist(self):
        return self._latent_distribution

    def is_image(self):
        return True

    def data_dim(self):
        return (1, self._image_size, self._image_size)

    def latent_dim(self):
        return self._latent_dim
    
    def parameter_dim(self):
        return None
    
    def _preprocess(self, img):
        return img
    
    def load_dataset(self, train):
        dataset = FixedBlobsManifold(self._args)
        l=len(dataset)
        print('original dataset length: %d' % l)
        train_length = int(self._split_ratio * l)
        test_length = l - train_length
        generator = torch.Generator().manual_seed(42)  # You can set any seed value
        self.train_dataset, self.test_dataset = random_split(dataset, [train_length, test_length], generator=generator)
        print('Train dataset len: %d' % len(self.train_dataset))
        if train:
            return self.train_dataset
        else:
            return self.test_dataset

class SyntheticDataset(Dataset):
    def __init__(self, args):
        super(SyntheticDataset, self).__init__()
        self.data, self.labels = self.create_dataset(args)
   
    def create_dataset(self, config):
        raise NotImplemented
        # return data, labels

    def log_prob(self, xs, ts):
        raise NotImplemented

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class FixedBlobsManifold(SyntheticDataset):
    def __init__(self, args):
        super().__init__(args)
    
    def get_the_gaussian_centers(self, seed, num_gaussians, std_range, img_size):
        random.seed(seed)
        guassians_info = []
        pairs = []
        for i in range(img_size):
            for j in range(img_size):
                pairs.append([i,j])
        
        #select the centers without replacement
        guassians_info = random.sample(pairs, k=num_gaussians)

        return guassians_info

    def create_dataset(self, args):
        print('Dataset Creation')
        num_samples = args.num_samples
        num_gaussians = args.num_gaussians #NEW
        std_range = args.std_range #NEW
        img_size = args.image_size #32
        seed = args.seed 

        centers_info = self.get_the_gaussian_centers(seed, num_gaussians, std_range, img_size)

        data = []
        for num in tqdm(range(num_samples)):
            img = torch.zeros(size=(img_size, img_size))
            for i in range(num_gaussians):
                x, y = centers_info[i]

                #paint the gaussians efficiently
                img = self.paint_the_gaussian(img, x, y, std_range)    
            
            #scale the image to [0, 1] range
            min_val, max_val = torch.min(img), torch.max(img)
            img -= min_val
            img /= max_val-min_val

            data.append(img.to(torch.float32).unsqueeze(0))
        
        data = torch.stack(data)
        return data, []
    
    def paint_the_gaussian(self, img, center_x, center_y, std_range):
        std = random.uniform(std_range[0], std_range[1])
        c = 1/(np.sqrt(2*np.pi)*std)
        new_img = torch.zeros_like(img)
        
        x = torch.tensor(np.arange(img.size(0)))
        y = torch.tensor(np.arange(img.size(1)))
        xx, yy = torch.meshgrid((x,y), indexing='ij')

        d = -1/(2*std**2)
        new_img = np.exp(d*((xx-center_x)**2+(yy-center_y)**2))
        new_img *= c
        img += new_img
        return img