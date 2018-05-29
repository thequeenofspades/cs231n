from __future__ import print_function, division
from scipy.io import loadmat
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from data_processing import StanfordDogsDataset, Flatten, mean_and_std
from baseline_model import model
import time
import matplotlib.pyplot as plt
import pickle as pkl
import shutil, os
import config

chkpoint_dir = 'results/baseline'
chkpoint_file = chkpoint_dir + '/checkpoint.pth.tar'
chkpoint_file_best = chkpoint_dir + '/model_best.pth.tar'

dtype = config.dtype
if config.use_GPU:
	print('Currently using CUDA device {}'.format(torch.cuda.current_device()))
	torch.cuda.device('cuda')
	model.cuda()

def save_checkpoint(state, is_best, filename=chkpoint_file):
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, chkpoint_file_best)

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, (0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[ 0.4742,  0.4323,  0.3795],
                             std=[1.0, 1.0, 1.0])
    ])

dataset = {x: StanfordDogsDataset(mat_file='lists/%s_list' % x,
                                    transform=data_transform)
			for x in ['train', 'test']}
dataloader = {x: DataLoader(dataset[x], batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
				for x in ['train', 'test']}

loss_fn = torch.nn.CrossEntropyLoss()

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if os.path.isfile(chkpoint_file):
	print("=> Loading checkpoint '{}'".format(chkpoint_file))
	checkpoint = torch.load(chkpoint_file)
	start_epoch = checkpoint['epoch']
	best_val_acc = checkpoint['best_val_acc']
	losses = checkpoint['losses']
	accs = checkpoint['accs']
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer'])
	print("=> Loaded checkpoint '{}' (epoch {}, best val accuracy {})".format(chkpoint_file, start_epoch, best_val_acc))
else:
	print("=> No checkpoint found at '{}'".format(chkpoint_file))
	start_epoch = 0
	best_val_acc = 0.0
	losses = {x: [] for x in ['train', 'test']}
	accs = {x: [] for x in ['train', 'test']}

for epoch in range(start_epoch, config.num_epochs):
	for phase in ['train','test']:
		tic = time.time()
		running_loss = 0.0
		running_corrects = 0
		for batch in dataloader[phase]:
			batch_images, batch_labels = batch['image'], batch['label']
			if config.use_GPU:
				batch_images = batch_images.cuda()
				batch_labels = batch_labels.cuda()
			with torch.set_grad_enabled(phase == 'train'):
				model.train(phase == 'train')
				y_pred = model(batch_images)
				loss = loss_fn(y_pred, batch_labels)
				running_loss += loss.item() * batch_images.shape[0]
				_, preds = torch.max(y_pred, 1)
				running_corrects += torch.sum(preds == batch_labels)

				optimizer.zero_grad()
				if phase == 'train':
					loss.backward()
					optimizer.step()
		loss = running_loss / len(dataset[phase])
		losses[phase].append(loss)
		acc = running_corrects.double() / len(dataset[phase])
		accs[phase].append(acc)
		print('{} {} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}s'.format(
		     	epoch+1, phase, loss, acc, time.time() - tic))

	is_best = (accs['test'][-1] >= best_val_acc)
	if is_best:
		best_val_acc = accs['test'][-1]
	state = {
		'epoch': epoch + 1,
		'state_dict': model.state_dict(),
		'best_val_acc': best_val_acc,
		'optimizer': optimizer.state_dict(),
		'losses': losses,
		'accs': accs
		}
	save_checkpoint(state, is_best)
	if (epoch + 1) % config.save_freq == 0:
		save_checkpoint(state, is_best, chkpoint_dir + '/checkpoint-%d.pth.tar' % (epoch+1))

# train_handle, = plt.plot(range(len(losses['train'])), losses['train'])
# test_handle, = plt.plot(range(len(losses['test'])), losses['test'])
# plt.title('Loss over %d epochs' % len(losses['train']))
# plt.xlabel('Epoch')
# plt.ylabel('CE loss')
# plt.legend([train_handle, test_handle], ['train', 'test'])
# plt.show()

# train_handle, = plt.plot(range(len(accs['train'])), accs['train'])
# test_handle, = plt.plot(range(len(accs['test'])), accs['test'])
# plt.title('Accuracy over %d epochs' % len(accs['train']))
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.legend([train_handle, test_handle], ['train', 'test'])
# plt.show()

# dataset = datasets.ImageFolder(root='Images', transform=data_transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# fig = plt.figure()

# for i in range(len(dataset['train'])):
#     sample = dataset['train'][i]

#     print(i, sample['image'].size, sample['label'])

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample['image'])

#     if i == 3:
#         plt.show()
#         break

# # Helper function to show a batch
# def show_dogs_batch(sample_batched):
#     """Show image with landmarks for a batch of samples."""
#     images_batch = sample_batched['image']
#     batch_size = len(images_batch)

#     grid = utils.make_grid(images_batch)
#     plt.imshow(grid.numpy().transpose((1, 2, 0)))

#     plt.title('Batch from dataloader')

# for i_batch, sample_batched in enumerate(dataloader['train']):
#     print(i_batch, sample_batched['image'].size(), sample_batched['label'])

#     # observe 4th batch and stop.
#     if i_batch == 3:
#         plt.figure()
#         show_dogs_batch(sample_batched)
#         plt.axis('off')
#         plt.ioff()
#         plt.show()
#         break