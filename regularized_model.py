from __future__ import print_function, division
import torch
from data_processing import Flatten
import config

flatten_size = (config.image_dim - 5 + 2 * 2) / 1 + 1
flatten_size = (flatten_size - 5 + 2 * 2) / 1 + 1
flatten_size = flatten_size / 2

model = torch.nn.Sequential(
			torch.nn.Conv2d(3, 16, 5, padding=2),
			torch.nn.ReLU(),
			torch.nn.LayerNorm([16, 32, 32]),
			torch.nn.Conv2d(16, 32, 5, padding=2),
			torch.nn.MaxPool2d(2),
			torch.nn.ReLU(),
			torch.nn.LayerNorm([32, 16, 16]),
			Flatten(),
			torch.nn.Linear(32*flatten_size*flatten_size, 400),
			torch.nn.ReLU(),
			torch.nn.Dropout(config.keep_prob),
			torch.nn.Linear(400, 200),
			torch.nn.ReLU(),
			torch.nn.Dropout(config.keep_prob),
			torch.nn.Linear(200, 120))