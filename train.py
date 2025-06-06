#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import time
import logging
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from medpy.metric import dc,hd95

from utils import DiceLoss
from test import inference
from dataset import ACDCdataset, RandomGenerator
from encoder import MTUNet
from model import LSTMSA

# train: nohup python -u train.py > train.log 2>&1 &
# test: python train.py --checkpoint ./checkpoint/best_model.pth --just_test --test_save_dir ./prediction

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, help="batch size")
parser.add_argument("--lr", default=0.0001, help="learning rate")
parser.add_argument("--max_epochs", default=100)
parser.add_argument("--img_size", default=224)
parser.add_argument("--save_path", default="./checkpoint/")
parser.add_argument("--n_gpu", default=1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--list_dir", default="./ACDC/lists_ACDC")
parser.add_argument("--root_dir", default="./ACDC/")
parser.add_argument("--volume_path", default="./ACDC/test")
parser.add_argument("--z_spacing", default=10)
parser.add_argument("--num_classes", default=4)
parser.add_argument('--test_save_dir', default=None) # suggest: './prediction'
parser.add_argument("--patches_size", default=16)
parser.add_argument("--n_skip", default=1)
parser.add_argument("--just_test", default=False, action="store_true")

# train: nohup python -u train.py --use_gradient_clipping > log/train_gradient_clipping.log 2>&1 &
# train: nohup python -u train.py --use_state_space_sequencing > log/train_state_space_sequencing.log 2>&1 &

# train: nohup python -u train.py --use_skip_connections_and_MLP > log/train_skip_connections_and_MLP.log 2>&1 &
# train: nohup python -u train.py --use_lambda_skip_connections > log/train_lambda_skip_connections.log 2>&1 &
# train: nohup python -u train.py --use_depth_dependent_residual_scaling > log/train_depth_dependent_residual_scaling.log 2>&1 &

parser.add_argument("--use_gradient_clipping", default=False, action="store_true")
parser.add_argument("--use_state_space_sequencing", default=False, action="store_true")
parser.add_argument("--use_skip_connections_and_MLP", default=False, action="store_true")
parser.add_argument("--use_lambda_skip_connections", default=False, action="store_true")
parser.add_argument("--use_depth_dependent_residual_scaling", default=False, action="store_true")

args = parser.parse_args()

use_gradient_clipping = args.use_gradient_clipping
use_state_space_sequencing = args.use_state_space_sequencing
use_skip_connections_and_MLP = args.use_skip_connections_and_MLP
use_lambda_skip_connections = args.use_lambda_skip_connections
use_depth_dependent_residual_scaling = args.use_depth_dependent_residual_scaling

encoder = MTUNet(64, use_skip_connections_and_MLP=use_skip_connections_and_MLP, use_depth_dependent_residual_scaling=use_depth_dependent_residual_scaling)
model = LSTMSA(num_classes=args.num_classes, encoder=encoder, use_state_space_sequencing=use_state_space_sequencing,
               use_lambda_skip_connections=use_lambda_skip_connections)

if args.checkpoint:
    model.load_state_dict(torch.load(args.checkpoint))

train_dataset = ACDCdataset(args.root_dir, args.list_dir, split="train", transform=
                                   transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
Train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
db_val=ACDCdataset(base_dir=args.root_dir, list_dir=args.list_dir, split="valid")
valloader=DataLoader(db_val, batch_size=1, shuffle=False)
db_test =ACDCdataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
testloader = DataLoader(db_test, batch_size=1, shuffle=False)

if args.n_gpu > 1:
    model = nn.DataParallel(model)

model = model.cuda()

if args.just_test:
    avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir)
    print("test avg_dsc: %f avg_hd: %f" % (avg_dcs, avg_hd))
    exit(0)
    
model.train()
ce_loss = CrossEntropyLoss()
dice_loss = DiceLoss(args.num_classes)
save_interval = args.n_skip  # int(max_epoch/6)

iterator = tqdm(range(0, args.max_epochs), ncols=70)
iter_num = 0

Loss = []
Test_Accuracy = []

logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s   %(levelname)s   %(message)s')

max_iterations = args.max_epochs * len(Train_loader)
base_lr = args.lr
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

start_time = time.time()
best_test_dsc, best_test_hd = 0, 100000.0
for epoch in iterator:
    model.train()
    train_loss = 0
    for i_batch, sampled_batch in enumerate(Train_loader):
        image_batch, prev_image_batch, next_image_batch, label_batch = sampled_batch["image"], sampled_batch["prev_image"], sampled_batch["next_image"], sampled_batch["label"]
        image_batch, label_batch = image_batch.type(torch.FloatTensor), label_batch.type(torch.FloatTensor)
        prev_image_batch, next_image_batch = prev_image_batch.type(torch.FloatTensor), next_image_batch.type(torch.FloatTensor)
        image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
        prev_image_batch, next_image_batch = prev_image_batch.cuda(), next_image_batch.cuda()
            
        outputs = model(prev_image_batch, image_batch, next_image_batch)
        
        loss_ce = ce_loss(outputs, label_batch[:].long())
        loss_dice = dice_loss(outputs, label_batch[:], softmax=True)
        loss = loss_dice * 0.5+ loss_ce * 0.5

        optimizer.zero_grad()
        loss.backward()
        
        if use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1

        logging.info('iteration %d : loss : %f lr_: %f' % (iter_num, loss.item(), lr_))
        train_loss += loss.item()
    Loss.append(train_loss/len(train_dataset))
    
    avg_dcs, avg_hd = inference(args, model, testloader, args.test_save_dir)
    print("test avg_dsc: %f avg_hd: %f" % (avg_dcs, avg_hd))
    if avg_dcs > best_test_dsc:
        best_test_dsc = avg_dcs
        # save model
        save_model_name = f"best_model.pth"
        torch.save(model.state_dict(), os.path.join(args.save_path, save_model_name))
    if avg_hd < best_test_hd:
        best_test_hd = avg_hd
    print("best test avg_dsc: %f best test avg_hd: %f" % (best_test_dsc, best_test_hd))

print("########################### Train Finished in %f minutes ###########################" % ((time.time() - start_time) // 60))
print("best test avg_dsc: %f best test avg_hd: %f" % (best_test_dsc, best_test_hd))