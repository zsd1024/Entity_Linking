mport numpy as np
import torch
from torch.autograd import Variable
from utils.scripts import *
import os
from tqdm import tqdm
from model.local_score_model import Local_Fai_score
from utils.data_loader_f import *
from data_process import Vocabulary
import torch.nn as nn
import datetime
import math
from Loss.selfloss import SelfLossFunc
import argparse
from torch.nn.init import kaiming_normal, uniform
import sys

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--epoch', default=30)
    arg_parser.add_argument('--LR', default=0.01)
    arg_parser.add_argument('--window_context', default=5)
    arg_parser.add_argument('--window_doc', default=100)
    arg_parser.add_argument('--window_body', default=100)
    arg_parser.add_argument('--window_title', default=17)
    arg_parser.add_argument('--filter_num', default=128)
    arg_parser.add_argument('--filter_window', default=5)
    arg_parser.add_argument('--embedding', default=300)
    arg_parser.add_argument('--lamda', default=0.01)
    #arg_parser.add_argument('--cuda_device', required=True, default=0)
    arg_parser.add_argument('--nohup', required=True, default="")
    arg_parser.add_argument('--batch', default=200)
    arg_parser.add_argument('--team', default=500)
    arg_parser.add_argument('--weight_decay', required=True, default=1e-5)
    arg_parser.add_argument('--embedding_finetune', default=1)
    arg_parser.add_argument('--local_model_loc', required=True)
    arg_parser.add_argument('--data_root', default="../data")

    arg_parser.add_argument('--char-channel-width', default=5, type=int)
    arg_parser.add_argument('--char-channel-size', default=200, type=int)
    arg_parser.add_argument('--dropout', default=0.2, type=float)
    arg_parser.add_argument('--hidden-size', default=100, type=int)
    arg_parser.add_argument('--word-dim', default=100, type=int)
    arg_parser.add_argument('--char-dim', default=8, type=int)

    args = arg_parser.parse_args()
    torch.manual_seed(1)
    EPOCH = int(args.epoch)
    LR = float(args.LR)
    WEIGHT_DECAY = float(args.weight_decay)
    WINDOW_CONTEXT = int(args.window_context)
    WINDOW_DOC = int(args.window_doc)
    WINDOW_BODY = int(args.window_body)
    WINDOW_TITLE = int(args.window_title)
    FILTER_NUM = int(args.filter_num)
    FILTER_WINDOW = int(args.filter_window)
    EMBEDDING = int(args.embedding)
    LAMDA = float(args.lamda)
    BATCH = int(args.batch)
    TEAM = int(args.team)
    FINETUNE = bool(int(args.embedding_finetune))
    LOCAL_MODEL_LOC = str(args.local_model_loc)
    ROOT = str(args.data_root)
    #torch.cuda.set_device(int(args.cuda_device))
    np.set_printoptions(threshold=sys.maxsize)

    print('Epoch num:              ' + str(EPOCH))
    print('Learning rate:          ' + str(LR))
    print('Weight decay:           ' + str(WEIGHT_DECAY))
    print('Context window:         ' + str(WINDOW_CONTEXT))
    print('Document window:        ' + str(WINDOW_DOC))
    print('Title window:           ' + str(WINDOW_TITLE))
    print('Body window:            ' + str(WINDOW_BODY))
    print('Filter number:          ' + str(FILTER_NUM))
    print('Filter window:          ' + str(FILTER_WINDOW))
    print('Embedding dim:          ' + str(EMBEDDING))
    print('Lambda:                 ' + str(LAMDA))
    print('Is finetune embedding:  ' + str(FINETUNE))
    print('Data root:              ' + str(ROOT))

    print("#######Data loading#######")
    data_loader_train = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY,
                                   val=False,
                                   test=False, shuffle=True, num_workers=0)
    data_loader_val = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY, val=True,
                                 test=False, shuffle=True, num_workers=0)
    data_loader_test = get_loader(ROOT, WINDOW_CONTEXT * 2, WINDOW_DOC, WINDOW_BODY, val=False,
                                  test=True, shuffle=True, num_workers=0)
    TrainFileNum = len(data_loader_train)
    print("Train data size:", len(data_loader_train))  # 337
    print("Dev data size:", len(data_loader_val))
    print("Test data size:", len(data_loader_test))
