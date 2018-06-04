from __future__ import print_function, division
import torch
from data_processing import Flatten
import config

flatten_size = (config.image_dim - 5 + 2 * 2) / 1 + 1
flatten_size = flatten_size / 2

model = torch.nn.Sequential(
			torch.nn.Conv2d(3, 16, 5, padding=2),
			torch.nn.MaxPool2d(2),
			torch.nn.ReLU(),
			Flatten(),
			torch.nn.Linear(16*flatten_size*flatten_size, 100),
			torch.nn.ReLU(),
			torch.nn.Linear(100, 120))