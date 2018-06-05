import matplotlib.pyplot as plt
import sys
import torch

chkpoint_file = sys.argv[1]
save_file = sys.argv[2]

checkpoint = torch.load(chkpoint_file, map_location={'cuda:0': 'cpu'})
start_epoch = checkpoint['epoch']
best_val_acc = checkpoint['best_val_acc']
losses = checkpoint['losses']
accs = checkpoint['accs']

print "Best val acc: %f" % best_val_acc

fig, ax = plt.subplots()

train_handle, = plt.plot(range(start_epoch), losses['train'])
test_handle, = plt.plot(range(start_epoch), losses['test'])
plt.title('Loss over %d epochs' % start_epoch)
plt.xlabel('Epoch')
plt.ylabel('CE loss')
plt.legend([train_handle, test_handle], ['train', 'test'])
fig.savefig(save_file + '-loss.png')

fig, ax = plt.subplots()

train_handle, = plt.plot(range(start_epoch), accs['train'])
test_handle, = plt.plot(range(start_epoch), accs['test'])
plt.title('Accuracy over %d epochs' % start_epoch)
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend([train_handle, test_handle], ['train', 'test'])
fig.savefig(save_file + '-acc.png')