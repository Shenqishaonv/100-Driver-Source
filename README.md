# 100-Driver-Source

This repository contains the code of our paper '100-Driver: A Large-scale, Diverse Dataset for Distracted Driver Classification'.

# Requirements

torch = 1.10.1

torchvision = 0.11.2

# Train a model
## 1. Prepare the data
Download the 100-Driver dataset from Google Drive or Baidu Yun.

Modify the data and data splits path in the function get_dataloader() in utils.py

## 2. Train

Please refer the .sh file in the scripts folder.


# Contact 

If you have any questions, feel free to contact us through email (<wjli007@mail.ustc.edu.cn>). Enjoy!

# BibTex
If you find this code or data useful, please consider citing our work.
    
    @InProceedings{100-Driver-2022,
    author    = {Wang, Jing and Li, Wengjing and Li, Fang and Zhang, Jun and Wu, Zhongcheng and Zhong, Zhun and Sebe, Nicu},
    title     = {100-Driver: A Large-scale, Diverse Dataset for Distracted Driver Classification},
    booktitle = {Under Review},
    year      = {2022}}
