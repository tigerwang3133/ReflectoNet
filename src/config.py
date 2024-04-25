from collections import defaultdict
import numpy as np
import os
import argparse
import pickle
from glob import glob
import random
from matplotlib import gridspec
import pandas as pd
import functools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing
import sys
import tqdm
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import math
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.models as models
from PIL import Image
from tensorboardX import SummaryWriter
from sklearn.metrics import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import *
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold
import cv2
from torch.autograd import Function
from torch.optim.lr_scheduler import ReduceLROnPlateau



# user defined variables
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-lr', default=0.1 , type=float)
parser.add_argument('-argumentation', default=1 , type=int)
parser.add_argument('-ml_n_epochs', default=60, type=int)
parser.add_argument('-lrdecay', default=0.9, type=int)
parser.add_argument('-hidden_dim', default=64, type=int)
parser.add_argument('-cross_val', default=True, type=bool)
parser.add_argument('-virus_type', default='reflection_5_19999', type=str)
parser.add_argument('-dl_n_epochs', default=64, type=int)
parser.add_argument('-batch_size', default=512, type=int)
parser.add_argument('-n_channels', default=10, type=int)
parser.add_argument('-n_classes', default=400, type=int)
parser.add_argument('-task', default='ml', type=str)
parser.add_argument('-exsistmodel', default='', type=str)
parser.add_argument('-noise', default=True, type=bool)
parser.add_argument('-kkrelation', default=False, type=bool)
parser.add_argument('-skip', default=False, type=bool)
opt = parser.parse_args()


# fix variables
TRAIN_PATH = '{}_plots/trainset'.format(opt.virus_type)
TEST_PATH = '{}_plots/testset'.format(opt.virus_type)
TEST_XTICK_PATH = '{}_plots/testset_xtick'.format(opt.virus_type)
RESULT_PATH = 'results/04-19-results'

    
if opt.virus_type == 'reflection_5_19999':
    ROOT = 'raw_data/reflection_5_19999/'
    PLOTS_PATH = 'reflection_5_19999'
    from csv import reader
    with open(ROOT+'dataset_list.csv', 'r',encoding="utf-8") as read_obj:
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            CLASSES=row
    CLASSES2IDX={k: v for v, k in enumerate(CLASSES)}



