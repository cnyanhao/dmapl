import random
import time
import sys
import argparse
import copy
import numpy as np
import os

import torch
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from dalib.modules.classifier import ImageClassifier
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy
from tools.transforms import train_transform_aug0, train_transform_center_crop
from tools.transforms import val_transform
from tools.lr_scheduler import StepwiseLR4


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


def main(args: argparse.Namespace):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    if args.center_crop:
        train_transform = train_transform_center_crop
    else:
        train_transform = train_transform_aug0

    # Data loading code
    dataset = datasets.__dict__[args.data]
    trainset = dataset(root=args.root, task=args.source, split='train', download=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)

    testset = dataset(root=args.root, task=args.source, split='test', download=True, transform=val_transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    backbone = models.__dict__[args.arch](pretrained=True)
    num_classes = trainset.num_classes
    classifier = ImageClassifier(backbone, num_classes).to(device)

    # define optimizer and lr scheduler
    optimizer = SGD(classifier.get_parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd, nesterov=True)
    lr_scheduler = StepwiseLR4(optimizer, max_iter=args.epochs*len(train_loader), init_lr=args.lr)

    # start training
    best_acc1 = 0.
    best_acc2 = 0.
    for epoch in range(args.epochs):

        # train for one epoch
        train(train_loader, classifier, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc1 = validate(test_loader, classifier, args)
        best_acc1 = max(acc1, best_acc1)

        if args.data == 'VisDA2017Split':
            acc2 = validate_per_class(test_loader, classifier, args)
            best_acc2 = max(acc2, best_acc2)       

        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1:
            best_model = copy.deepcopy(classifier.state_dict())   

    print("best_acc1 = {:3.1f}".format(best_acc1))
    if args.data == 'VisDA2017Split':
        print("best_acc2 = {:3.1f}".format(best_acc2))

    # save source model
    save_name = f'source_models/{args.data}_{args.source}_{args.arch}.pth'
    dir = os.path.dirname(save_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(best_model, save_name)


def train(train_loader: DataLoader, model: ImageClassifier, optimizer: SGD,
          lr_scheduler: StepwiseLR4, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses = AverageMeter('Loss', ':3.2f')
    cls_accs = AverageMeter('Cls Acc', ':3.1f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, cls_accs],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    print(lr_scheduler.get_lr())

    end = time.time()
    for i, (images, target) in enumerate(train_loader):

        lr_scheduler.step()

        if images.size(0) <= 1:
            continue

        images = images.to(device)
        target = target.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        y_s, _ = model(images)

        loss = F.cross_entropy(y_s, target)

        cls_acc = accuracy(y_s, target)[0]

        losses.update(loss.item(), images.size(0))
        cls_accs.update(cls_acc.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            loss = F.cross_entropy(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def validate_per_class(val_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time,],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    predicts = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output, _ = model(images)
            predict = output.argmax(dim=1)
            
            predicts.append(predict)
            targets.append(target)
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        predicts = torch.cat(predicts)
        targets = torch.cat(targets)
        
        matrix = confusion_matrix(targets.cpu().float(), predicts.cpu().float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)

        print(f' * Acc@1: {aacc}, Accs: {acc}')

    return aacc


if __name__ == '__main__':
    architecture_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
    )
    dataset_names = sorted(
        name for name in datasets.__dict__
        if not name.startswith("__") and callable(datasets.__dict__[name])
    )

    parser = argparse.ArgumentParser(description='PyTorch Domain Adaptation')
    parser.add_argument('root', metavar='DIR',
                        help='root path of dataset')
    parser.add_argument('-d', '--data', metavar='DATA', default='Office31',
                        help='dataset: ' + ' | '.join(dataset_names) +
                             ' (default: Office31)')
    parser.add_argument('-s', '--source', help='source domain(s)')
    parser.add_argument('-t', '--target', help='target domain(s)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=architecture_names,
                        help='backbone architecture: ' +
                             ' | '.join(architecture_names) +
                             ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.001, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('-p', '--print-freq', default=100, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--center_crop', default=False, action='store_true')

    args = parser.parse_args()
    print(args)
    main(args)
