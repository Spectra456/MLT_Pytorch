'''Train CIFAR10 with PyTorch.'''
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import random
from tqdm import tqdm
from itertools import cycle

import os
import argparse

from dataset import *
from model import *

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(43)

def accuracy_top_k(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    output = output.to('cpu')
    target = target.to('cpu')
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res[0].item()

def batch_handler(inputs, targets,output):
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs[output], targets)
    loss.backward()
    optimizer.step()

    return outputs,loss

def train(epoch, iterator_fashion):
    print('\nEpoch: %d' % epoch)
    model.train()

    train_loss = 0
    total_acc = 0
    counter = 0
    iterator_cifar = iter(train_loader_cifar)

    test_progress_bar = tqdm(range(len(iterator_cifar)))
    test_progress_bar.set_description('Train')

    for batch_idx in test_progress_bar:
        if (bool(random.getrandbits(1)) == True):
            # training of Fashion RGB
            index = 0
            counter +=1

            inputs, targets = next(iterator_fashion)
            outputs, loss = batch_handler(inputs,targets,index)
            train_loss += loss.item()
            targets = targets.to('cpu')
            total_acc += accuracy_top_k(outputs[index], targets)

        # We making it, because cifar10 bigger than fashion RGB. 
        if (bool(random.getrandbits(1)) == True):
            # training of Cifar10
            index = 1
            counter +=1

            inputs, targets = next(iterator_cifar)
            outputs, loss = batch_handler(inputs,targets,index)
            train_loss += loss.item()
            targets = targets.to('cpu')
            total_acc += accuracy_top_k(outputs[index], targets)

         # Adding meticrs to tensorboards scalars
    writer.add_scalar('Top-1 Accuracy/train', total_acc/counter, epoch)
    writer.add_scalar('Loss/train', train_loss/counter, epoch)

    print('Top-1: {} %, Loss: {}'.format(round(total_acc/counter,2), train_loss/counter))
    

def test(epoch, index):
    global best_acc
    model.eval()

    test_loss = 0
    total_acc = 0

    if index == 0:
        test_progress_bar = tqdm(test_loader_fashion)
    else:
        test_progress_bar = tqdm(test_loader_cifar)

    test_progress_bar.set_description('Test')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs[index], targets)

            test_loss += loss.item()
            total_acc += accuracy_top_k(outputs[index], targets)

    # Adding meticrs to tensorboards scalars
    writer.add_scalar('Top-1 Accuracy/{} test'.format(datasets[index]), round((total_acc)/(batch_idx+1),2), epoch)
    writer.add_scalar('Loss/{} test'.format(datasets[index]), (test_loss)/(batch_idx+1), epoch)

    print('{} dataset, Top-1: {} %, Loss: {}'.format(datasets[index], round((total_acc)/(batch_idx+1),2), (test_loss)/(batch_idx+1)))

    # Save checkpoint.
    acc = (total_acc)/(batch_idx+1)
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc



parser = argparse.ArgumentParser(description='PyTorch Multi-Task Learning')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume training from the best checkpoint')
parser.add_argument('--epochs', type=int, default=200)  
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size', type=tuple, default=(32, 32), help='input size of image ')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu' #Choosing cuda by default device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
datasets = ['FashionRGB', 'Cifar10'] # List of datasets

transform_train = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.RandomCrop(args.img_size[0], padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


train_set_cifar = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
test_set_cifar = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)

train_set_fashion = ImageDataset('{}/train.json'.format('dataset'), transform_train)
test_set_fashion = ImageDataset('{}/val.json'.format('dataset'), transform_test)


train_loader_cifar = torch.utils.data.DataLoader(train_set_cifar, batch_size=128, shuffle=True, num_workers=os.cpu_count())
test_loader_cifar = torch.utils.data.DataLoader(test_set_cifar, batch_size=128, shuffle=False, num_workers=os.cpu_count())

train_loader_fashion = torch.utils.data.DataLoader(train_set_fashion, batch_size=128, shuffle=True, num_workers=os.cpu_count())
test_loader_fashion = torch.utils.data.DataLoader(test_set_fashion, batch_size=128, shuffle=False, num_workers=os.cpu_count())

# Writer for TensorBoard
writer = SummaryWriter(flush_secs=13, log_dir='logs/MLT')

# Model
model = EfficientNetB0()
model = model.to(device)

# Ahh, sorry for DataParallel. I want to use Apex for FP16 and multi-gpu, but i don't have enough time...
if device == 'cuda':
    model = torch.nn.DataParallel(model)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

# Maybe we can use here something like Focal Loss(because dataset is unbalanced) modified for multi-task learning, but i don't have enough time :((
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# Start training

for epoch in range(start_epoch, start_epoch+args.epochs):
    # Using cycle because cifar have 50k images and Fashion RGB have 5k. So in every iteration we taking 1 batch from first and from second.
    iterator = cycle(train_loader_fashion)

    train(epoch, iterator)
    test(epoch, 0)
    test(epoch, 1)

    #scheduler.step()
