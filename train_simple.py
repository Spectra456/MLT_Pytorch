import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
from tqdm import tqdm

from dataset import *
from model import *

torch.backends.cudnn.deterministic = True # Making some stuff for  computations deterministic
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)


def accuracy_top_k(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    pred = output.topk(maxk, 1, True, True)[1]
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res[0].item()

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()

    train_loss = 0
    total_acc = 0

    train_progress_bar = tqdm(train_loader)
    train_progress_bar.set_description('Train')

    for batch_idx, (inputs, targets) in enumerate(train_progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs[dataset_index], targets)
        loss.backward()
        optimizer.step()

        # Calculating metrics
        train_loss += loss.item()
        total_acc += accuracy_top_k(outputs[dataset_index], targets)

    # Adding meticrs to tensorboards scalars
    writer.add_scalar('Top-1 Accuracy/train', round((total_acc/(batch_idx+1)),2), epoch)
    writer.add_scalar('Loss/train', train_loss/(batch_idx+1), epoch)

    print('Top-1: {} %, Loss: {}'.format(round((total_acc/(batch_idx+1)),2), train_loss/(batch_idx)))
    

def test(epoch):
    global best_acc
    model.eval()

    test_loss = 0
    total_acc = 0

    test_progress_bar = tqdm(test_loader)
    test_progress_bar.set_description('Test')

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs[dataset_index], targets)

            test_loss += loss.item()
            total_acc += accuracy_top_k(outputs[dataset_index], targets)

    # Adding meticrs to tensorboards scalars
    writer.add_scalar('Top-1 Accuracy/test', round((total_acc/(batch_idx+1)),2), epoch)
    writer.add_scalar('Loss/test', test_loss/(batch_idx+1), epoch)

    print('Top-1: {} %, Loss: {}'.format(round(((total_acc)/(batch_idx+1)),2), (test_loss)/(batch_idx+1)))

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

"""============================================================================================================================================================================="""
parser = argparse.ArgumentParser(description='PyTorch FashionRGB and CIFAR10 training for test task')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume training from the best checkpoint')
parser.add_argument('--epochs', type=int, default=200)  
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--img_size', type=tuple, default=(32, 32), help='input size of image ')
parser.add_argument('--dataset', type=str, default='dataset', help='CIFAR10 or FashionRGB')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu' #Choosing cuda by default device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

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

# Choosing dataset by setting index, it's also using for getting output from model
if args.dataset == 'CIFAR10':
    dataset_index = 1

if args.dataset == 'FashionRGB':
    dataset_index = 0


if dataset_index == 1: 
    train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform_test)
else:
    train_set = ImageDataset('dataset/train.json', transform_train)
    test_set = ImageDataset('dataset/val.json', transform_test)


train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())

writer = SummaryWriter(flush_secs=13, log_dir='logs/{}'.format(args.dataset))

model = EfficientNetB0()
model = model.to(device)

# Ahh, sorry for DataParallel. I want to use Apex for FP16 and multi-gpu, but i don't have enough time...
if device == 'cuda':
    model = torch.nn.DataParallel(model)

# Start training from checkpoint
if args.resume:
    # Load checkpoint.
    print('Resuming from checkpoint...')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1) # 30 for Cifar, 60 for Fashion

# Start training
for epoch in range(start_epoch, start_epoch+args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
