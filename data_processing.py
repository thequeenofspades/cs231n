from __future__ import print_function, division
from scipy.io import loadmat
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from xml.etree import cElementTree as ET
from PIL import Image
import config

dtype = config.dtype

class StanfordDogsDataset(Dataset):
    """Stanford Dogs dataset."""

    def __init__(self, mat_file, transform=None):
        """
        Args:
            mat_file (string): Path to the mat file containing the file list.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.file_list = loadmat(mat_file)
        self.transform = transform

    def __len__(self):
    	return len(self.file_list['labels'])

    def __getitem__(self, idx):
        img_name = os.path.join('Images',
                                self.file_list['file_list'][idx][0][0])
        image = Image.open(img_name)
        image = image.convert('RGB')
        label = self.file_list['labels'][idx][0] - 1
        annotation_name = os.path.join('Annotation',
        							self.file_list['annotation_list'][idx][0][0])
        bndbox = ET.parse(annotation_name).find('object/bndbox')
        image = image.crop((int(bndbox.find('xmin').text),
        					int(bndbox.find('ymin').text),
        					int(bndbox.find('xmax').text),
        					int(bndbox.find('ymax').text)))
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample

class Flatten(torch.nn.Module):
	def forward(self, x):
		N = x.shape[0]
		return x.view(N, -1)

def mean_and_std(dataset):
	mean = torch.tensor([0., 0., 0.])
	for i in range(len(dataset)):
		sample = dataset[i]
		image, label = sample['image'], sample['label']
		print(i, image.shape)
		image = image.reshape((3, -1))
		image_mean = image.mean(1)
		mean += image_mean
	mean /= len(dataset)
	print(mean)