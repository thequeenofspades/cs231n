from __future__ import print_function, division
import torch
import torchvision.models as models
from data_processing import Flatten
import config

model = models.resnet18()