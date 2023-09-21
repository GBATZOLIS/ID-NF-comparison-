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

class SquaresManifoldSimulator(BaseSimulator):
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
        dataset = FixedSquaresManifold(self._args)
        l=len(dataset)
        print('original dataset length: %d' % l)
        train_length = int(self._split_ratio * l)
        test_length = l - train_length
        self.train_dataset, self.test_dataset = random_split(dataset, [train_length, test_length])
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

class FixedSquaresManifold(SyntheticDataset):
    def __init__(self, args):
        super().__init__(args)
    
    def get_the_squares(self, seed, num_squares, square_range, img_size):
        random.seed(seed)
        squares_info = []
        for _ in range(num_squares):
            side = random.choice(square_range)
            start = (side+1)//2
            finish = img_size - (side+1)//2
            x = random.choice(np.arange(start, finish))
            y = random.choice(np.arange(start, finish))
            squares_info.append([x, y, side])

        return squares_info

    def create_dataset(self, args):
        print('Dataset Creation')
        num_samples = args.num_samples
        num_squares = args.num_squares #10
        square_range = args.square_range #[3, 5]
        img_size = args.image_size #32
        seed = args.seed 

        squares_info = self.get_the_squares(seed, num_squares, square_range, img_size)

        data = []
        for num in tqdm(range(num_samples)):
            img = torch.zeros(size=(img_size, img_size))
            for i in range(num_squares):
                x, y, side = squares_info[i]
                img = self.paint_the_square(img, x, y, side)   
            data.append(img.to(torch.float32).unsqueeze(0))
        
        data = torch.stack(data)
        return data, []
    
    def paint_the_square(self, img, center_x, center_y, side):
        c = random.random()
        for i in range(side):
            for j in range(side):
                img[center_x - ((side+1)//2 - 1) + i, center_y - ((side+1)//2 - 1) + j]+=c
        return img