""" helper function

author wenjing
"""
import os
import sys
import re
import datetime
import numpy
import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as model
from dataset import DataSet
from models.ghostnet import ghostnet


def get_torch_network(args):
    #resnet
    if args.net == 'resnet18':
        if args.pretrain:
            net = model.resnet18(pretrained=True)
        else:
            net = model.resnet18(pretrained=False)
    elif args.net == 'resnet34':
        if args.pretrain:
            net = model.resnet34(pretrained=True)
        else:
            net = model.resnet34(pretrained=False)
    elif args.net == 'resnet50':
        if args.pretrain:
            net = model.resnet50(pretrained=True)
        else:
            net = model.resnet50(pretrained=False)
    elif args.net == 'resnet101':
        if args.pretrain:
            net = model.resnet101(pretrained=True)
        else:
            net = model.resnet101(pretrained=False)
    elif args.net == 'resnet152':
        if args.pretrain:
            net = model.resnet152(pretrained=True)
        else:
            net = model.resnet152(pretrained=False)
    elif args.net == 'wide_resnet50':
        if args.pretrain:
            net = model.wide_resnet50_2(pretrained=True)
        else:
            net = model.wide_resnet50_2(pretrained=False)
    elif args.net == 'wide_resnet101':
        if args.pretrain:
            net = model.wide_resnet101_2(pretrained=True)
        else:
            net = model.wide_resnet101_2(pretrained=False)
    #VGG
    elif args.net == 'vgg19bn':
        if args.pretrain:
            net = model.vgg19_bn(pretrained=True)
        else:
            net = model.vgg19_bn(pretrained=False)
    elif args.net == 'vgg19':
        if args.pretrain:
            net = model.vgg19(pretrained=True)
        else:
            net = model.vgg19(pretrained=False)
    elif args.net == 'vgg16bn':
        if args.pretrain:
            net = model.vgg16_bn(pretrained=True)
        else:
            net = model.vgg16_bn(pretrained=False)
    elif args.net == 'vgg16':
        if args.pretrain:
            net = model.vgg16(pretrained=True)
        else:
            net = model.vgg16(pretrained=False)
    elif args.net == 'vgg13bn':
        if args.pretrain:
            net = model.vgg13_bn(pretrained=True)
        else:
            net = model.vgg13_bn(pretrained=False)
    elif args.net == 'vgg13':
        if args.pretrain:
            net = model.vgg13(pretrained=True)
        else:
            net = model.vgg13(pretrained=False)
    elif args.net == 'vgg11bn':
        if args.pretrain:
            net = model.vgg11_bn(pretrained=True)
        else:
            net = model.vgg11_bn(pretrained=False)
    elif args.net == 'vgg11':
        if args.pretrain:
            net = model.vgg11(pretrained=True)
        else:
            net = model.vgg11(pretrained=False)
    #densenet
    elif args.net == 'dense121':
        if args.pretrain:
            net = model.densenet121(pretrained=True)
        else:
            net = model.densenet121(pretrained=False)
    elif args.net == 'dense161':
        if args.pretrain:
            net = model.densenet161(pretrained=True)
        else:
            net = model.densenet161(pretrained=False)
    elif args.net == 'dense169':
        if args.pretrain:
            net = model.densenet169(pretrained=True)
        else:
            net = model.densenet169(pretrained=False)
    elif args.net == 'dense201':
        if args.pretrain:
            net = model.densenet201(pretrained=True)
        else:
            net = model.densenet201(pretrained=False)
    #inceptionv3
    elif args.net == 'inceptionv3':
        if args.pretrain:
            net = model.inception_v3(pretrained=True, aux_logits=False)
        else:
            net = model.inception_v3(pretrained=False, aux_logits=False)
    elif args.net == 'mobilenet_v2':
        if args.pretrain:
            net = model.mobilenet_v2(pretrained=True)
        else:
            net = model.mobilenet_v2(pretrained=False)
    elif args.net == 'mobilenetv3_large':
        if args.pretrain:
            net = model.mobilenet_v3_large(pretrained=True)
        else:
            net = model.mobilenet_v3_large(pretrained=False)
    elif args.net == 'mobilenetv3_small':
        if args.pretrain:
            net = model.mobilenet_v3_small(pretrained=True)
        else:
            net = model.mobilenet_v3_small(pretrained=False)
    elif args.net == 'shufflenetv2_1_0':
        if args.pretrain:
            net = model.shufflenet_v2_x1_0(pretrained=True)
        else:
            net = model.shufflenet_v2_x1_0(pretrained=False)
    elif args.net == 'squeezenet1_0':
        if args.pretrain:
            net = model.squeezenet1_0(pretrained=True)
        else:
            net = model.squeezenet1_0(pretrained=False)
    elif args.net == 'efficientnet_b0':
        if args.pretrain:
            net = model.efficientnet_b0(pretrained=True)
        else:
            net = model.efficientnet_b0(pretrained=False)  
    elif args.net == 'ghost1_0':
        if args.pretrain:
            net = ghostnet(pretrained=True)
        else:
            net = ghostnet(pretrained=False)       
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
#
#    if args.gpu: #use_gpu
#        net = net.cuda()
    return net


def modify_output(args, net):
    classifier_zoo = ['dense121', 'dense161', 'dense201', 'ghost1_0']
    vgg_zoo = ['vgg19bn','vgg16bn']
    Inception_zoo = ['inceptionv3', 'inceptionv4']
    mobile_zoo = [ 'mobilenetv2', 'mobilenetv3_large', 'mobilenetv3_small','efficientnet_b0']
    squeeze_zoo = ['squeezenet1_0', 'squeezenet1_1']
    if args.net in classifier_zoo:
        channel_in = net.classifier.in_features
        net.classifier = nn.Linear(channel_in, args.num_class)
    elif args.net in mobile_zoo:
#        print('0000000000')
#        print(net.classifier[-1])
        channel = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(channel, args.num_class)
    elif args.net in Inception_zoo :
        channel_in = net.linear.in_features
        net.fc = nn.Linear(channel_in, args.num_class)
    elif args.net in vgg_zoo:
        net.classifier[-1] = nn.Linear(4096, args.num_class)
    elif args.net in squeeze_zoo:
        net.classifier[1] = nn.Conv2d(512, args.num_class, kernel_size=1)
    else:
        channel_in = net.fc.in_features
        net.fc = nn.Linear(channel_in, args.num_class)
    return net


def get_training_dataloader(dataset, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        dataset:
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """
    training = dataset
    training_loader = DataLoader(
        training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return training_loader

def get_test_dataloader(dataset, batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        dataset:
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: test_data_loader:torch dataloader object
    """
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    test = dataset
    test_loader = DataLoader(
        test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)

    return test_loader


def get_dataloader(dataset):
    """ return training dataloader
    Args:
        dataset: the name of the dataset
    Returns:
        loader: the path of the train and val data
        loadertxt: the txt file from the train and val set
    """
#Traditional setting for Cam1,2,3,4; if other settings are required, just add it in another elif         
    if dataset == 'pic-day-cam1':
        trainloader = './data/100-driver/Day_RGB/Cam1'
        trainloadertxt = './data-splits/Traditional-setting/Day/Cam1/D1_train.txt'
        valloader = './data/100-driver/Day_RGB/Cam1'
        valloadertxt = './data-splits/Traditional-setting/Day/Cam1/D1_val.txt'
    elif dataset == 'pic-day-cam2':
        trainloader = './data/100-driver/Day_RGB/Cam2'
        trainloadertxt = './data-splits/Traditional-setting/Day/Cam2/D2_train.txt'
        valloader = './data/100-driver/Day_RGB/Cam2'
        valloadertxt = './data-splits/Traditional-setting/Day/Cam2/D2_train.txt'
    elif dataset == 'pic-day-cam3':
        trainloader = './data/100-driver/Day_RGB/Cam3'
        trainloadertxt = './data-splits/Traditional-setting/Day/Cam3/D3_train.txt'
        valloader = './data/100-driver/Day_RGB/Cam3'
        valloadertxt = './data-splits/Traditional-setting/Day/Cam3/D3_val.txt'
    elif dataset == 'pic-day-cam4':
        trainloader = './data/100-driver/Day_RGB/Cam4'
        trainloadertxt = './data-splits/Traditional-setting/Day/Cam4/D4_train.txt'
        valloader = './data/100-driver/Day_RGB/Cam4'
        valloadertxt = './data-splits/Traditional-setting/Day/Cam4/D4_val.txt'
    else:
        print('the dataset is not available ')
    return trainloader, trainloadertxt, valloader, valloadertxt

def compute_mean_std(dataset):
    """compute the mean and std of the input dataset
    Args:
        training_dataset which derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([dataset[i][1][:, :, 0] for i in range(len(dataset))])
    data_g = numpy.dstack([dataset[i][1][:, :, 1] for i in range(len(dataset))])
    data_b = numpy.dstack([dataset[i][1][:, :, 2] for i in range(len(dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


def get_mean_std(dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        dataset: the name of the dataset

    Returns:
        a list contains mean, std value of entire dataset which is obtained by the function compute_mean_std

    """
    day = ['statefarm','aucv2-camera1','aucv2-camera2', 'aucv1', '3mdad-cam1',
           '3mdad-cam2','3mdad-cam1-wash','3mdad-cam2-wash','pic-day-all','pic-day-cam1',
           'pic-day-cam2','pic-day-cam3', 'pic-day-cam4','pic-xiandai-cam1','pic-xiandai-cam2',
           'pic-xiandai-cam3','pic-xiandai-cam4','pic-day-car-cam1','pic-day-car-cam2',
           'pic-day-car-cam3','pic-day-car-cam4','3mdad-day']

    mdad_night = ['3mdad-cam1-night','3mdad-cam2-night',]

    pic_night = ['pic-night-all', 'pic-night-cam1','pic-night-cam2',
                 'pic-night-cam3','pic-night-cam4','pic-night-car-cam1',
                 'pic-night-car-cam2','pic-night-car-cam3', 'pic-night-car-cam4' ]
    if dataset in day:
        mean=[.5, .5, .5]
        std=[0.229, 0.224, 0.225]
    elif dataset in mdad_night:
        mean = [0.046468433, 0.046468433, 0.046468433]
        std = [0.051598676, 0.051598676, 0.051598676]
    elif dataset in pic_night:
        mean = [0.29414198, 0.3019768, 0.29021993]
        std = [0.24205828, 0.24205923, 0.24205303]
    else:
        print('the dataset is not available ')
    return mean, std

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [f for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f, fmt))
    return folders[-1]

def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]

def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
       raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch

def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]

def unnormalized(img):
    t_mean = torch.FloatTensor(mean).view(3,1,1).expand(3, 224, 224)
    t_std = torch.FloatTensor(std).view(3,1,1).expand(3, 224, 224)
    img = img * t_std + t_mean     # unnormalize
    img = img
    trans = transforms.ToPILImage()
    img = trans(img)
    return img

    
if __name__ == '__main__':

    trainloader, trainloadertxt, valloader, valloadertxt = get_dataloader('3mdad-cam1-night')

    train_datasets = DataSet(trainloader, trainloadertxt, flag ='train')#get data

    #val_datasets = CarDateSet(valloader, valloadertxt, flag = 'val')#

    mean,std = compute_mean_std(train_datasets)
    print(mean, std)
