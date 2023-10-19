# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author wenjing
"""

import os
import sys
import argparse
import time
from datetime import datetime
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import DataSet
from conf import settings
from utils import get_training_dataloader, get_test_dataloader, WarmUpLR, \
    get_torch_network, get_dataloader, get_mean_std, modify_output,get_train_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epoch):

    start = time.time()
    net.train()
    loss_train = 0.0
    acc_train = 0.0
    correct_prediction = 0.0
    # total = 0.0
    for batch_index, (images, labels) in enumerate(training_loader):
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()
            
        outputs = net(images)
        optimizer.zero_grad()
        loss = loss_function(outputs, labels)
        loss_train += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        loss.backward()
        optimizer.step()
        correct_prediction += (predicted == labels).sum().item()
        # total += labels.size(0)
        #####
        if epoch <= args.warm:
            warmup_scheduler.step()
        #######
        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))
    # total_batch = len(training_loader)
    train_loss = loss_train /len(training_loader)
    train_acc = correct_prediction / len(train_datasets)

    finish = time.time()
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
# validation
    if epoch % 1 == 0:
        start = time.time()
        net.eval()
        test_loss = 0.0 # cost function error
        correct = 0.0
    
        for batch_idx, (images, labels) in enumerate(val_loader):

            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
            with torch.no_grad():
                outputs = net(images)
                loss = loss_function(outputs, labels)
                test_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum()

        finish = time.time()
        test_loss = test_loss / len(val_loader)
        test_acc = correct.float() / len(val_datasets)
   
        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary())
        print('Evaluating Network.....')
        print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
            test_loss,
            test_acc,
            finish - start
        ))
        print()
    
    wandb.log({"train Acc": train_acc}, commit=False)
    wandb.log({"train Loss": train_loss},commit=False)
    
    wandb.log({"val Acc": test_acc}, commit=False)
    wandb.log({ "val Loss": test_loss})

    return test_acc


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=2, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-pretrain', default=True, help='wether the pretrain model is used')
    parser.add_argument("-dataset", default='pic-day-cam1', type=str)
    parser.add_argument("-num_class", default=22, type=int)

    args = parser.parse_args()

# bulid the backbone model
    net = get_torch_network(args)
    net = modify_output(args, net)
    if args.pretrain:
        print("load pretrain model successful")

    for name,parameters in net.named_parameters():
        print(name,':',parameters.size())
    

# the config for wandb output
    config = dict(
    architecture = args.net,
    dataset_id = args.dataset,
    batch_size = args.b,
    pretrain = args.pretrain,
    lr = 'StepLR50'
     )
     
    wandb.init(project="pic-car-157", entity = 'wenjing', config=config, name=args.dataset + '_' + args.net)


    if args.gpu:
        net = net.cuda()

# prepare the data
    trainloader, trainloadertxt, valloader, valloadertxt = get_dataloader(args.dataset)
    mean, std = get_mean_std(args.dataset)
    train_datasets = DataSet(trainloader, trainloadertxt, mean, std, flag ='train')#get data
    val_datasets = DataSet(valloader, valloadertxt, mean, std, flag = 'val')#
    training_loader = get_training_dataloader(
        dataset = train_datasets,
        num_workers=4,
        batch_size=args.b,
        shuffle=True  
    )
    val_loader = get_test_dataloader(
        dataset = val_datasets,
        num_workers=4,
        batch_size=args.b,
        shuffle=True  
    )

# define loss function
    loss_function = nn.CrossEntropyLoss()

# prepare the optimizer
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

# set detail of the checkpoint storage path
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH,args.dataset,'split_by_driver', args.net)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

# start training the model
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH+1):
        if epoch > args.warm:
            train_scheduler.step(epoch)
        if args.resume:
            if epoch <= resume_epoch:
                continue
        acc = train(epoch)

        # save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            print("best model! save...")
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

