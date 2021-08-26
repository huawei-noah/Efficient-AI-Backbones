'''
2019.07.24 Changed details for LegoNet
           Huawei Technologies Co., Ltd. <foss@huawei.com>
'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import random
import time
import os
import argparse
import logging
import glob
import sys

from vgg import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.1, type=float)
parser.add_argument('--batch_size', type = int, default = 128)
parser.add_argument('--test_batch_size', type = int, default = 1024)
parser.add_argument('--arch', default='lego_vgg16', type = str, help='architecture')
parser.add_argument('--gpu', default=[0], type = list)
parser.add_argument('--dataset', default = 'c10', type = str)
parser.add_argument('--log_interval', default = 100, type = int)
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--kernel_size', type=int, default=3, help='3 for resnet like model, 1 for mobilelike model')
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--seed', type = int, default = 2)
parser.add_argument('--n_split', type = int, default = 2)
parser.add_argument('--epochs', type = int, default = 400)
parser.add_argument('--warmup', type = int, default = 10)
parser.add_argument('--n_lego', type = float, default = 0.5)
parser.add_argument('--balance_weight', type = float, default = 1e-4)
args = parser.parse_args()
args.save = 'Lego-{}-NS{}-NL{}-{}'.format(args.dataset, args.n_split, args.n_lego, time.strftime("%Y%m%d-%H%M%S"))
print(args)

create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(g) for g in args.gpu])

print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'c10':
    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=False, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
    n_classes = 10

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory = True, num_workers=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory = True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.arch == 'lego_vgg16':
    model = lego_vgg16(args.arch, args.n_split, args.n_lego, n_classes)
    print(model)
    
logging.info("args = %s", args)
model = model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad ], lr=args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

def train(epoch, args):
    logging.info('\nEpoch: %d, Learning rate: %f', epoch, scheduler.get_lr()[0])
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    end = time.time()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        data_time = time.time()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        model.copy_grad(args.balance_weight)
        optimizer.step()
        if epoch < args.warmup:
            model.copy_grad(args.balance_weight)
        optimizer.zero_grad()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_loss += loss.item()
        model_time = time.time()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: %d Process: %d Total: %d    Loss: %.06f    Data Time: %.03f s    Model Time: %.03f s    Memory %.03fMB', 
                epoch, batch_idx * len(inputs), len(trainloader.dataset), loss.item(), data_time - end, model_time - data_time, count_memory(model))
        end = time.time()

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct, test_loss / len(testloader)

if __name__ == '__main__':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    cudnn.enabled = True
    torch.manual_seed(args.seed)
    
    max_correct = 0
    for epoch in range(args.epochs):
        if epoch == args.warmup:
            optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad and 'combination' not in n], lr=args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warmup)
        scheduler.step()
        train(epoch, args)
        correct, loss = test(epoch)
        if correct > max_correct:
            max_correct = correct
            torch.save(model, os.path.join(args.save, 'weights.pth'))
        logging.info('Epoch %d Correct: %d, Max Correct %d, Loss %.06f', epoch, correct, max_correct, loss)
