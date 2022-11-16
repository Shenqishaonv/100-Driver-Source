#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author wenjing
"""

import argparse
from dataset import DataSet
import torch
from utils import get_test_dataloader, get_torch_network, modify_output
import time

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    parser.add_argument('-pretrain', type=bool, default=False, help='use pretrain model or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-num_class', type=int, default=22, help='the number of the classes')
    parser.add_argument('-model_root', type=int, default=22, help='the weights file you want to test')
    args = parser.parse_args()
    net = get_torch_network(args)

#load model from the checkpoint path
    net = modify_output(args, net)
    
    net = net.cuda()
 
    checkpoint = torch.load(args.model_root)

    net.eval()


# prepare test data
    val_datasets = DataSet('/home/liwenjing/Backbones/data/PIC/day/angle2_rgb',
                           '/home/liwenjing/Backbones/data/PIC/day-xiandai/cam2/test.txt', flag='val')
    test_loader = get_test_dataloader(
        dataset = val_datasets,
        num_workers=4,
        batch_size=args.b,
        shuffle=True  
    )


# start testing
    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                
            time0 = time.time()
            output = net(image)
            time1 = time.time()
            timed = time1 - time0
            print('time',timed)

            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()


    print()
    print("Top 1 err: ", 1 - correct_1 / len(test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
