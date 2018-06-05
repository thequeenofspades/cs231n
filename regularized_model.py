from __future__ import print_function, division
import torch
from data_processing import Flatten
import config

flatten_size = (config.image_dim - 5 + 2 * 2) / 1 + 1
flatten_size = int(flatten_size / 2)
layer_norm_1 = int(flatten_size)
flatten_size = (flatten_size - 5 + 2 * 2) / 1 + 1
flatten_size = int(flatten_size / 2)
layer_norm_2 = int(flatten_size)
flatten_size = (flatten_size - 5 + 2 * 2) / 1 + 1
flatten_size = int(flatten_size / 2)
layer_norm_3 = int(flatten_size)

layer_norm_size_1 = [16, layer_norm_1, layer_norm_1]
layer_norm_size_2 = [32, layer_norm_2, layer_norm_2]
layer_norm_size_3 = [64, layer_norm_3, layer_norm_3]

model = torch.nn.Sequential(
			torch.nn.Conv2d(3, 16, 5, padding=2),
			torch.nn.MaxPool2d(2),
			torch.nn.ReLU(),
			torch.nn.LayerNorm(layer_norm_size_1),
			torch.nn.Conv2d(16, 32, 5, padding=2),
			torch.nn.MaxPool2d(2),
			torch.nn.ReLU(),
			torch.nn.LayerNorm(layer_norm_size_2),
			torch.nn.Conv2d(32, 64, 5, padding=2),
			torch.nn.MaxPool2d(2),
			torch.nn.ReLU(),
			torch.nn.LayerNorm(layer_norm_size_3),
			Flatten(),
			torch.nn.Linear(64*flatten_size*flatten_size, 400),
			torch.nn.ReLU(),
			torch.nn.Dropout(config.keep_prob),
			torch.nn.Linear(400, 200),
			torch.nn.ReLU(),
			torch.nn.Dropout(config.keep_prob),
			torch.nn.Linear(200, 120))