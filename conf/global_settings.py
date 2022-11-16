""" configurations for this project

author Wenjing
"""
import os
from datetime import datetime

#directory to save weights file
CHECKPOINT_PATH = './checkpoint'

#total training epoches
EPOCH = 100
MILESTONES = [40, 80]

#initial learning rate
#INIT_LR = 0.1

DATE_FORMAT = '%A_%d_%B_%Y_%Hh_%Mm_%Ss'
#time of we run the script
TIME_NOW = datetime.now().strftime(DATE_FORMAT)

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10








