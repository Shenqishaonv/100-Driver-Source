# 100-Driver-Source

This repository contains the code of our paper '100-Driver: A Large-scale, Diverse Dataset for Distracted Driver Classification' https://drive.google.com/file/d/1JhLdk-feblXi_pepF7GTb-6vHwtrbxSr/view.

# Requirements

torch = 1.10.1

torchvision = 0.11.2

# Train a model
## 1. Prepare the data
Download the 100-Driver dataset from [here](https://100-driver.github.io/).

Put the data in the ./data folder

If you want to change the data and data splits path, please refer to the function get_dataloader() in utils.py

## 2. Train

Please refer to the train.sh file in the scripts folder.

## 3. Test

Please refer to the test.sh file in the scripts folder.

# Contact 

If you have any questions, feel free to contact us through email (<wjli007@mail.ustc.edu.cn>). Enjoy!

# BibTex
If you find this code or data useful, please consider citing our work.
   
    @article{100-Driver,
    author    = {Wang Jing, Li Wengjing, Li Fang, Zhang Jun, Wu Zhongcheng, Zhong Zhun and Sebe Nicu},
    title     = {100-Driver: A Large-scale, Diverse Dataset for Distracted Driver Classification},
    journal={IEEE Transactions on Intelligent Transportation Systems},
    year      = {2023}
    publisher={IEEE}}
