# coding:utf8
import os
from PIL import Image
from torch.utils import data
from torchvision import transforms


class DataSet(data.Dataset):

    def __init__(self, root, lists, mean, std, flag):
        with open(lists, 'r') as f:
            lines = f.readlines()
        imgs = []
        labels = []

        for line in lines:
            imgs.append(os.path.join(root, line.split()[1]))
            labels.append(int(line.split()[2]))

        self.imgs = imgs
        self.labels = labels

        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])


        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(30),
            transforms.ToTensor(),

            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing()
        ])
        if flag == 'train':

            self.transforms = transform_train
        else:
            self.transforms = transform_test

    def __getitem__(self, index):

        img_path = self.imgs[index]
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        label = self.labels[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':

    dataset = DataSet('./data/night/cam1', './data/night/cam1_list/train.txt', flag='train')
    img, label = dataset[0]
    for img, label in dataset:
        print(img.size(), img.float().mean(), label)
