import random
import time
import sys
import argparse
import numpy as np

import torch
import torch.nn.parallel
from torch.optim import SGD
import torch.utils.data
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

sys.path.append('.')
from dalib.modules.classifier import ImageClassifier
from dalib.adaptation.sfda import DatasetIndex, SubsetImagesPlabels, PseudoPartialLabelDisambiguation
import dalib.vision.datasets as datasets
import dalib.vision.models as models
from tools.utils import AverageMeter, ProgressMeter, accuracy
from tools.transforms import train_transform_aug0, train_transform_center_crop
from tools.transforms import val_transform
from tools.lr_scheduler import StepwiseLR4 as StepwiseLR


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
    trainset = dataset(root=args.root, task=args.target, split='train', download=True, transform=train_transform)
    # train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)

    valset = dataset(root=args.root, task=args.target, split='train', download=True, transform=val_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    testset = dataset(root=args.root, task=args.target, split='test', download=True, transform=val_transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    # create source model
    backbone = models.__dict__[args.arch](pretrained=False)
    num_classes = testset.num_classes
    trg_model = ImageClassifier(backbone, num_classes).to(device)
    trg_model.load_state_dict(torch.load(f'source_models/{args.data}_{args.source}_{args.arch}.pth'))
    
    # divide source and target training set
    src_indices, src_plabels, trg_indices = divide_surrogate_src_trg(trg_model, val_loader, args)

    srcset = SubsetImagesPlabels(trainset, src_indices, src_plabels)
    src_loader = DataLoader(srcset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    trgset = Subset(trainset, trg_indices)
    trgset_idx = DatasetIndex(trgset)
    trg_loader = DataLoader(trgset_idx, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

    # trg soft lables updater
    ppl_updater = PseudoPartialLabelDisambiguation(len(trgset), trg_model.features_dim, num_classes, device, args.coeff_cent, args.coeff_prob)

    # define optimizer and lr scheduler
    param_group = trg_model.get_parameters()
    optimizer = SGD(param_group, args.lr, momentum=args.momentum, 
                    weight_decay=args.weight_decay, nesterov=True)
    lr_scheduler = StepwiseLR(optimizer, max_iter=args.epochs*len(trg_loader), init_lr=args.lr)

    
    # start training
    print("=> start training")
    best_acc1 = 0.
    best_acc2 = 0.
    for epoch in range(args.epochs):

        # train for one epoch
        train(trg_loader, trg_model, src_loader, ppl_updater, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        acc2 = validate(test_loader, trg_model, args)
        best_acc2 = max(acc2, best_acc2)
        if args.data == 'VisDA2017Split':
            acc1 = validate_per_class(test_loader, trg_model, args)
            best_acc1 = max(acc1, best_acc1)

    print("best_acc2 = {:3.1f}".format(best_acc2))
    if args.data == 'VisDA2017Split':
        print("best_acc1 = {:3.1f}".format(best_acc1))


def train(trg_loader: DataLoader, trg_model: ImageClassifier, src_loader: DataLoader, ppl_updater: PseudoPartialLabelDisambiguation,
        optimizer: SGD, lr_scheduler: StepwiseLR, epoch: int, args: argparse.Namespace):
    
    losses = AverageMeter('Loss', ':6.2f')
    progress = ProgressMeter(
        len(trg_loader),
        [losses,],
        prefix="Epoch: [{}]".format(epoch))

    trg_model.train()
    # domain_adv.train()

    src_iter = iter(src_loader)

    for i, (images_t, _, data_idx) in enumerate(trg_loader):
        
        lr_scheduler.step()
        
        if images_t.size(0) <= 1:
            continue
        images_t = images_t.to(device)

        try:
            images_s, labels_s = next(src_iter)
        except:
            src_iter = iter(src_loader)
            images_s, labels_s = next(src_iter)
        images_s = images_s.to(device)
        labels_s = labels_s.to(device)

        data_idx = data_idx.to(device)

        # compute output
        y_s, f_s = trg_model(images_s)
        y_t, f_t = trg_model(images_t)

        loss = 0

        # src ce loss
        src_ce_loss = F.cross_entropy(y_s, labels_s)
        loss += src_ce_loss * args.param_sce

        # trg ce loss
        plabels_t = y_t.argmax(dim=1)
        prob_t = ppl_updater.update_parameters(f_s, labels_s, f_t, plabels_t, data_idx)
        tce_loss = - (prob_t * F.log_softmax(y_t, dim=1)).sum(dim=1).mean()
        loss += tce_loss

        losses.update(loss.item(), images_t.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(test_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
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


def validate_per_class(test_loader: DataLoader, model: ImageClassifier, args: argparse.Namespace) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(test_loader),
        [batch_time,],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    predicts = []
    targets = []
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
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


def divide_surrogate_src_trg(model, val_loader, args):
    model.eval()

    probs = torch.empty(0, dtype=torch.float32)
    plabels = torch.empty(0, dtype=torch.int64)

    with torch.no_grad():
        for _, (images, _) in enumerate(val_loader):
            
            if images.size(0) <= 1:
                continue
            
            images = images.to(device)
            
            y, _ = model(images)
            p = F.softmax(y, dim=1)
            prob, plabel = p.max(dim=1)
            probs = torch.cat((probs, prob.cpu()))
            plabels = torch.cat((plabels, plabel.cpu()))
            
        # src
        src_indices = torch.where(probs > args.prob_th)[0]
        src_plabels = plabels[src_indices]

        trg_indices = torch.where(probs <= args.prob_th)[0]

    return src_indices, src_plabels, trg_indices


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
    parser.add_argument('--wd', '--weight-decay',default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--center_crop', default=False, action='store_true')
    parser.add_argument('--prob_th', default=0.9, type=float)
    parser.add_argument('--param_sce', default=1., type=float)
    parser.add_argument('--coeff_cent', default=0.9, type=float)
    parser.add_argument('--coeff_prob', default=0.9, type=float)

    args = parser.parse_args()
    print(args)
    main(args)

