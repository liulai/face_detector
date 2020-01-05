import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import argparse
from torchvision import datasets, transforms
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#from data_myself20191221 import get_train_test_set
from data_myself import get_train_test_set
import copy
import datetime
import pandas as pd
import cv2
import PIL.Image as Image

import torchvision

###image height and weight
train_boarder = 112


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Linear2Conv(nn.Module):
    def __init__(self, height, weight):
        super(Linear2Conv, self).__init__()
        self.height = height
        self.weight = weight

    def forward(self, x):
        x = x.view(x.size(0), -1, self.height, self.weight)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(8, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(16, 24, 3, 1),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.Conv2d(24, 24, 3, 1),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(24, 40, 3, 1, 1),
            nn.BatchNorm2d(40),
            nn.PReLU()
        )

        self.layer2_1 = nn.Sequential(
            nn.Conv2d(40, 40, 3, 1, 1),
            nn.BatchNorm2d(40),
            nn.PReLU(),
            Flatten(),
            nn.Linear(40 * 4 * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 2)
        )

        self.layer2_2 = nn.Sequential(
            nn.Conv2d(40, 80, 3, 1, 1),
            nn.BatchNorm2d(80),
            nn.PReLU(),
            Flatten(),
            nn.Linear(80 * 4 * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 42)
        )

    def forward(self, x):
        x = self.layer1(x)
        x1 = self.layer2_1(x)
        x2 = self.layer2_2(x)

        return x1, x2


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 8, 5, 2),
            nn.BatchNorm2d(8),
            nn.PReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(8, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, 3, 1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(16, 24, 3, 1),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.Conv2d(24, 24, 3, 1),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(24, 40, 3, 1, 1),
            nn.BatchNorm2d(40),
            nn.PReLU()
        )

        self.layer2_1 = nn.Sequential(
            nn.Conv2d(40, 40, 3, 1, 1),
            nn.BatchNorm2d(40),
            nn.PReLU(),
            Flatten(),
            nn.Linear(40 * 4 * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 2)
        )

        self.layer2_2 = nn.Sequential(
            nn.Conv2d(40, 80, 3, 1, 1),
            nn.BatchNorm2d(80),
            nn.PReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(80, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            # Flatten(),
            # nn.Linear(80 * 4 * 4, 80 * 4 * 4),
            # nn.PReLU(),
            # nn.Linear(80 * 4 * 4, 80 * 4 * 4),
            # nn.PReLU(),
            # Linear2Conv(4, 4),
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(),
            nn.ConvTranspose2d(256, 80, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(80),
            nn.PReLU(),
            nn.ConvTranspose2d(80, 40, 3, 2, 1, output_padding=1),
            nn.BatchNorm2d(40),
            nn.PReLU(),
            nn.ConvTranspose2d(40, 32, 3, 2, 2),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.ConvTranspose2d(32, 24, 3, 2),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.ConvTranspose2d(24, 22, 3, 2, output_padding=1),
            nn.BatchNorm2d(22),
            nn.PReLU(),
            nn.ConvTranspose2d(22, 21, 3, 2, 1, output_padding=1)

        )

    def forward(self, x):
        x = self.layer1(x)
        x1 = self.layer2_1(x)
        x1 = nn.Softmax(dim=1)(x1)
        x2 = self.layer2_2(x)
        # print('x2.size:',x2.size())

        return x1, x2


class Net_Res(nn.Module):
    def __init__(self):
        super(Net_Res, self).__init__()
        model_res101 = torchvision.models.resnet101(pretrained=False, progress=True)
        self.res = nn.Sequential(*list(model_res101.children())[:-2])
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 2)
        )

        self.layer2_2 = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.PReLU(),
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.PReLU(),
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 42)
        )

    def forward(self, x):
        x = self.res(x)
        x1 = self.layer2_1(x)
        x2 = self.layer2_2(x)

        return x1, x2


class Net_Res18(nn.Module):
    def __init__(self):
        super(Net_Res18, self).__init__()
        model_res18 = torchvision.models.resnet18(pretrained=False, progress=True)
        self.res = nn.Sequential(*list(model_res18.children())[:-2])
        self.layer2_1 = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 2)
        )

        self.layer2_2 = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.PReLU(),
            Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.PReLU(),
            nn.Linear(128, 128),
            nn.PReLU(),
            nn.Linear(128, 42)
        )

    def forward(self, x):
        x = self.res(x)
        x1 = self.layer2_1(x)
        x1 = nn.Softmax(dim=1)(x1)
        x2 = self.layer2_2(x)

        return x1, x2


def train(args, model, optimer, critierion_face, critierion_landmarks, train_loader, val_loader, device, heatmap=False):
    print('args:', args)
    print('device:', device)
    print('batch_size:', args.batch_size)
    train_face_list = []
    train_ldm_list = []
    train_list = []
    train_face_per_list = []
    train_ldm_per_list = []
    train_per_list = []

    val_face_list = []
    val_ldm_list = []
    val_list = []
    val_face_per_list = []
    val_ldm_per_list = []
    val_per_list = []

    train_acc_list = []
    train_acc_per_list = []
    train_acc_neg_list = []
    train_acc_neg_per_list = []
    train_acc_pos_list = []
    train_acc_pos_per_list = []

    val_acc_list = []
    val_acc_per_list = []
    val_acc_neg_list = []
    val_acc_neg_per_list = []
    val_acc_pos_list = []
    val_acc_pos_per_list = []

    ##based on heatmap compute
    ldms_list = []
    ldms_v_list = []

    for i in range(args.epochs):

        train_loss = 0.0
        val_loss = 0.0
        train_face_loss = 0.0
        val_face_loss = 0.0
        train_ldm_loss = 0.0
        val_ldm_loss = 0.0

        train_acc = 0.0
        train_acc_neg = 0.0
        train_acc_pos = 0.0
        train_acc_neg_total = 0
        train_acc_pos_total = 0

        val_acc = 0.0
        val_acc_neg = 0.0
        val_acc_pos = 0.0
        val_acc_neg_total = 0
        val_acc_pos_total = 0

        ##based on heatmap compute
        ldms_err = 0.0
        ldms_err_v = 0.0

        ##count number of landmarks
        train_num = 0
        val_num = 0
        model.train()
        for id, sample in enumerate(train_loader):
            optimer.zero_grad()

            inputs = sample['image'].to(device)
            # print('inputs.size:', inputs.size())
            targets_face = sample['face'].to(device)
            if heatmap:
                targets_landmarks = sample['landmarks_heatmap'].to(device)
            else:
                targets_landmarks = sample['landmarks'].to(device)

            outputs_face, outputs_landmarks = model(inputs)

            #             with torchsnooper.snoop():
            loss_face = critierion_face(outputs_face, targets_face)

            train_per_neg = torch.zeros(outputs_face.size(0), dtype=torch.long).to(device)
            train_per_pos = torch.ones(outputs_face.size(0), dtype=torch.long).to(device)

            train_per_neg_num = torch.sum(targets_face == 0)
            train_per_pos_num = torch.sum(targets_face == 1)

            train_acc_neg_total += train_per_neg_num.item()
            train_acc_pos_total += train_per_pos_num.item()

            train_acc_per_neg_num = 1.0 * torch.sum(
                (torch.argmax(outputs_face, dim=1) == targets_face) & (targets_face == train_per_neg),
                dtype=torch.float)

            train_acc_per_pos_num = 1.0 * torch.sum(
                (torch.argmax(outputs_face, dim=1) == targets_face) & (targets_face == train_per_pos),
                dtype=torch.float)

            train_acc_per_neg = train_acc_per_neg_num / train_per_neg_num
            train_acc_per_pos = train_acc_per_pos_num / train_per_pos_num
            train_acc_per = (train_acc_per_neg_num + train_acc_per_pos_num) / targets_face.size(0)

            outputs_landmarks = outputs_landmarks[targets_face == 1]
            targets_landmarks = targets_landmarks[targets_face == 1]

            if targets_landmarks.size(0) == 0:
                loss_landmarks = torch.tensor(0.0, device=device)
            else:
                loss_landmarks = critierion_landmarks(outputs_landmarks, targets_landmarks)

            loss = args.loss_rate * loss_face + (1 - args.loss_rate) * loss_landmarks

            loss.backward()
            optimer.step()

            if heatmap:
                ## based on heatmap compute original landmarks loss for values and predict values
                coordinates = [torch.stack((torch.argmax(torch.max(item, dim=2)[0], dim=1),
                                            torch.argmax(torch.max(item, dim=1)[0], dim=1))).permute(1, 0).reshape(
                    -1, ).tolist()
                               for item in targets_landmarks]
                coordinates = torch.tensor(coordinates, dtype=torch.float)

                coordinates_pre = [torch.stack((torch.argmax(torch.max(item, dim=2)[0], dim=1),
                                                torch.argmax(torch.max(item, dim=1)[0], dim=1))).permute(1, 0).reshape(
                    -1, ).tolist()
                                   for item in outputs_landmarks]
                coordinates_pre = torch.tensor(coordinates_pre, dtype=torch.float)

                ldms_err += torch.mean((coordinates - coordinates_pre) ** 2).item() * outputs_face.size(0)

            train_per_list.append(loss.item())
            train_face_per_list.append(loss_face.item())
            train_ldm_per_list.append(loss_landmarks.item())

            train_acc_per_list.append(train_acc_per.item())
            train_acc_neg_per_list.append(train_acc_per_neg.item())
            train_acc_pos_per_list.append(train_acc_per_pos.item())

            train_face_loss += loss_face.item() * outputs_face.size(0)
            train_ldm_loss += loss_landmarks.item() * outputs_face.size(0)
            train_loss += loss.item() * outputs_face.size(0)
            if targets_landmarks.size(0) != 0:
                train_num += outputs_face.size(0)

            train_acc += (train_acc_per_neg_num.item() + train_acc_per_pos_num.item())
            train_acc_neg += train_acc_per_neg_num.item()
            train_acc_pos += train_acc_per_pos_num.item()

            loss_v_0 = -1.0
            loss_face_v_0 = -1.0
            loss_landmarks_v_0 = -1.0
            val_acc_per_v_0 = -1.0
            val_acc_per_neg_v_0, val_acc_per_pos_v_0 = -1.0, -1.0
            for sample_v_0 in val_loader:
                inputs_v_0 = sample_v_0['image'].to(device)
                targets_face_v_0 = sample_v_0['face'].to(device)
                if heatmap:
                    targets_landmarks_v_0 = sample_v_0['landmarks_heatmap'].to(device)
                else:
                    targets_landmarks_v_0 = sample_v_0['landmarks'].to(device)
                outputs_face_v_0, outputs_landmarks_v_0 = model(inputs_v_0)

                loss_face_v_0 = critierion_face(outputs_face_v_0, targets_face_v_0)

                val_per_neg_v_0 = torch.zeros(outputs_face_v_0.size(0), dtype=torch.long).to(device)
                val_per_pos_v_0 = torch.ones(outputs_face_v_0.size(0), dtype=torch.long).to(device)

                val_per_neg_num_v_0 = torch.sum(targets_face_v_0 == 0)
                val_per_pos_num_v_0 = torch.sum(targets_face_v_0 == 1)

                val_acc_per_neg_num_v_0 = 1.0 * torch.sum(
                    (torch.argmax(outputs_face_v_0, dim=1) == targets_face_v_0) & (targets_face_v_0 == val_per_neg_v_0),
                    dtype=torch.float)

                val_acc_per_pos_num_v_0 = 1.0 * torch.sum(
                    (torch.argmax(outputs_face_v_0, dim=1) == targets_face_v_0) & (targets_face_v_0 == val_per_pos_v_0),
                    dtype=torch.float)

                val_acc_per_neg_v_0 = val_acc_per_neg_num_v_0 / val_per_neg_num_v_0
                val_acc_per_pos_v_0 = val_acc_per_pos_num_v_0 / val_per_pos_num_v_0
                val_acc_per_v_0 = (val_acc_per_neg_num_v_0 + val_acc_per_pos_num_v_0) / len(targets_face_v_0)

                outputs_landmarks_v_0 = outputs_landmarks_v_0[targets_face_v_0 == 1]
                targets_landmarks_v_0 = targets_landmarks_v_0[targets_face_v_0 == 1]
                loss_landmarks_v_0 = critierion_landmarks(outputs_landmarks_v_0, targets_landmarks_v_0)
                loss_v_0 = args.loss_rate * loss_face_v_0 + (1 - args.loss_rate) * loss_landmarks_v_0

                val_per_list.append(loss_v_0.item())
                val_face_per_list.append(loss_face_v_0.item())
                val_ldm_per_list.append(loss_landmarks_v_0.item())

                val_acc_per_list.append(val_acc_per_v_0.item())
                val_acc_neg_per_list.append(val_acc_per_neg_v_0.item())
                val_acc_pos_per_list.append(val_acc_per_pos_v_0.item())

                break

            if id % args.log_interval == 0:
                print(
                    'Epoch:{} [{}/{}]{:.2f}% per train loss(* face ldm):{:.3f} {:.3f} {:.3f}  '
                    'val loss(...):{:.3f} {:.3f} {:.3f} train/val(* pos neg) acc:[{:.3f} {:.3f} {:.3f}] '
                    '[{:.3f} {:.3f} {:.3f}]'.format(
                        i, id * args.batch_size, len(train_loader.dataset), 100.0 * id / len(train_loader),
                        loss.item(), loss_face.item(), loss_landmarks.item(),
                        loss_v_0.item(), loss_face_v_0.item(), loss_landmarks_v_0.item(),
                        train_acc_per, train_acc_per_pos, train_acc_per_neg, val_acc_per_v_0, val_acc_per_pos_v_0,
                        val_acc_per_neg_v_0
                    ))

        train_list.append(train_loss / len(train_loader.dataset))
        train_face_list.append(train_face_loss / len(train_loader.dataset))
        # train_ldm_list.append(train_ldm_loss / len(train_loader.dataset))
        train_ldm_list.append(train_ldm_loss / train_num)

        train_acc_list.append(train_acc / len(train_loader.dataset))
        train_acc_neg_list.append(train_acc_neg / train_acc_neg_total)
        train_acc_pos_list.append(train_acc_pos / train_acc_pos_total)

        #### based on heatmap compute
        ldms_list.append(ldms_err / train_num)
        # train_acc_neg_list.append(train_acc_neg/torch.sum((train_loader.dataset)['face']==0))

        starttime = datetime.datetime.now()
        model.eval()
        with torch.no_grad():
            for sample_v in val_loader:
                inputs_v = sample_v['image'].to(device)
                targets_face_v = sample_v['face'].to(device)
                if heatmap:
                    targets_landmarks_v = sample_v['landmarks_heatmap'].to(device)
                else:
                    targets_landmarks_v = sample_v['landmarks'].to(device)

                outputs_face_v, outputs_landmarks_v = model(inputs_v)

                loss_face_v = critierion_face(outputs_face_v, targets_face_v)

                val_per_neg = torch.zeros(outputs_face_v.size(0), dtype=torch.long).to(device)
                val_per_pos = torch.ones(outputs_face_v.size(0), dtype=torch.long).to(device)

                val_per_neg_num = torch.sum(targets_face_v == 0)
                val_per_pos_num = torch.sum(targets_face_v == 1)

                val_acc_neg_total += val_per_neg_num.item()
                val_acc_pos_total += val_per_pos_num.item()

                val_acc_per_neg_num = 1.0 * torch.sum(
                    (torch.argmax(outputs_face_v, dim=1) == targets_face_v) & (targets_face_v == val_per_neg),
                    dtype=torch.float)

                val_acc_per_pos_num = 1.0 * torch.sum(
                    (torch.argmax(outputs_face_v, dim=1) == targets_face_v) & (targets_face_v == val_per_pos),
                    dtype=torch.float)

                val_acc += (val_acc_per_neg_num.item() + val_acc_per_pos_num.item())
                val_acc_neg += val_acc_per_neg_num.item()
                val_acc_pos += val_acc_per_pos_num.item()

                outputs_landmarks_v = outputs_landmarks_v[targets_face_v == 1]
                targets_landmarks_v = targets_landmarks_v[targets_face_v == 1]

                if targets_landmarks_v.size(0) == 0:
                    loss_landmarks_v = torch.tensor(0.0, device=device)
                else:
                    loss_landmarks_v = critierion_landmarks(outputs_landmarks_v, targets_landmarks_v)
                # loss_landmarks_v = critierion_landmarks(outputs_landmarks_v, targets_landmarks_v)
                loss_v = args.loss_rate * loss_face_v + (1 - args.loss_rate) * loss_landmarks_v

                if heatmap:
                    if targets_landmarks_v.size(0) != 0:
                        ##based on heatmap compute original landmarks loss for values and predict values
                        coordinates_v = [torch.stack((torch.argmax(torch.max(item, dim=2)[0], dim=1),
                                                      torch.argmax(torch.max(item, dim=1)[0], dim=1))).permute(1,
                                                                                                               0).reshape(
                            -1, ).tolist()
                                         for item in targets_landmarks_v]
                        coordinates_v = torch.tensor(coordinates_v, dtype=torch.float)

                        coordinates_pre_v = [torch.stack((torch.argmax(torch.max(item, dim=2)[0], dim=1),
                                                          torch.argmax(torch.max(item, dim=1)[0], dim=1))).permute(1,
                                                                                                                   0).reshape(
                            -1, ).tolist()
                                             for item in outputs_landmarks_v]
                        coordinates_pre_v = torch.tensor(coordinates_pre_v, dtype=torch.float)

                        ldms_err_v += torch.mean((coordinates_v - coordinates_pre_v) ** 2).item() * outputs_face_v.size(
                            0)

                val_loss += loss_v.item() * outputs_face_v.size(0)
                val_face_loss += loss_face_v.item() * outputs_face_v.size(0)
                val_ldm_loss += loss_landmarks_v.item() * outputs_face_v.size(0)
                if targets_landmarks_v.size(0) != 0:
                    val_num += outputs_face_v.size(0)

        val_list.append(val_loss / len(val_loader.dataset))
        val_face_list.append(val_face_loss / len(val_loader.dataset))
        # val_ldm_list.append(val_ldm_loss / len(val_loader.dataset))

        val_ldm_list.append(val_ldm_loss / val_num)
        val_acc_list.append(val_acc / len(val_loader.dataset) * 1.0)
        # train_acc_list.append(train_acc / len(train_loader.dataset))
        val_acc_neg_list.append(val_acc_neg / val_acc_neg_total)
        val_acc_pos_list.append(val_acc_pos / val_acc_pos_total)

        ####based on heatmap compute
        ldms_v_list.append(ldms_err_v / val_num)

        if heatmap:
            print(
                'TRAIN LOSS(* face ldm):{:.3f} {:.3f} {:.3f} {:.3f}'
                ' VAL LOSS(* face ldm):{:.3f} {:.3f} {:.3f} {:.3f} T/V(* pos neg) acc:[{:.3f} {:.3f} {:.3f}] '
                '[{:.3f} {:.3f} {:.3f}] runtime(s):{}'.format(
                    train_loss / len(train_loader.dataset),
                    train_face_loss / len(train_loader.dataset),
                    train_ldm_loss / len(train_loader.dataset),
                    ldms_err / train_num,
                    val_loss / len(val_loader.dataset),
                    val_face_loss / len(val_loader.dataset),
                    val_ldm_loss / val_num,
                    ldms_err_v / val_num,
                    train_acc / len(train_loader.dataset) * 1.0,
                    train_acc_pos / train_acc_pos_total,
                    train_acc_neg / train_acc_neg_total,
                    val_acc / len(val_loader.dataset) * 1.0,
                    val_acc_pos / val_acc_pos_total,
                    val_acc_neg / val_acc_neg_total,
                    (datetime.datetime.now() - starttime).total_seconds()))
        else:
            print(
                'TRAIN LOSS(* face ldm):{:.3f} {:.3f} {:.3f} '
                'VAL LOSS(* face ldm):{:.3f} {:.3f} {:.3f} T/V(* pos neg) acc:[{:.3f} {:.3f} {:.3f}] '
                '[{:.3f} {:.3f} {:.3f}] runtime(s):{}'.format(
                    train_loss / len(train_loader.dataset),
                    train_face_loss / len(train_loader.dataset),
                    train_ldm_loss / len(train_loader.dataset),
                    val_loss / len(val_loader.dataset),
                    val_face_loss / len(val_loader.dataset),
                    val_ldm_loss / val_num,
                    train_acc / len(train_loader.dataset) * 1.0,
                    train_acc_pos / train_acc_pos_total,
                    train_acc_neg / train_acc_neg_total,
                    val_acc / len(val_loader.dataset) * 1.0,
                    val_acc_pos / val_acc_pos_total,
                    val_acc_neg / val_acc_neg_total,
                    (datetime.datetime.now() - starttime).total_seconds()))

    time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    if (args.save_model):
        if (not os.path.exists(args.save_directory)):
            os.mkdir(args.save_directory)
        torch.save(model.state_dict(), os.path.join(args.save_directory, 'detector_%s_%s.pt' % (time_str, args.phase)))
        torch.save(model, os.path.join(args.save_directory, 'detector_%s_%s_model.pt' % (time_str, args.phase)))

        print('the parameters(model) state dict is saved, path:{}'.format(
            os.path.join(args.save_directory, 'detector_%s_%s.pt' % (time_str, args.phase))))

    # save loss_list
    if (not os.path.exists(args.csv_dir)):
        os.mkdir(args.csv_dir)

        # save Loss_list
    train_list = train_list + [np.nan for i in range(len(train_face_per_list) - len(train_list))]
    train_face_list = train_face_list + [np.nan for i in range(len(train_face_per_list) - len(train_face_list))]
    train_ldm_list = train_ldm_list + [np.nan for i in range(len(train_face_per_list) - len(train_ldm_list))]
    val_list = val_list + [np.nan for i in range(len(train_face_per_list) - len(val_list))]
    val_face_list = val_face_list + [np.nan for i in range(len(train_face_per_list) - len(val_face_list))]
    val_ldm_list = val_ldm_list + [np.nan for i in range(len(train_face_per_list) - len(val_ldm_list))]

    train_acc_list = train_acc_list + [np.nan for i in range(len(train_face_per_list) - len(train_acc_list))]
    val_acc_list = val_acc_list + [np.nan for i in range(len(train_face_per_list) - len(val_acc_list))]

    train_acc_neg_list = train_acc_neg_list + [np.nan for i in
                                               range(len(train_face_per_list) - len(train_acc_neg_list))]
    train_acc_pos_list = train_acc_pos_list + [np.nan for i in
                                               range(len(train_face_per_list) - len(train_acc_pos_list))]

    val_acc_neg_list = val_acc_neg_list + [np.nan for i in
                                           range(len(train_face_per_list) - len(val_acc_neg_list))]
    val_acc_pos_list = val_acc_pos_list + [np.nan for i in
                                           range(len(train_face_per_list) - len(val_acc_pos_list))]

    ldms_list = ldms_list + [np.nan for i in range(len(train_face_per_list) - len(ldms_list))]
    ldms_v_list = ldms_v_list + [np.nan for i in range(len(train_face_per_list) - len(ldms_v_list))]

    loss_list = {'train': train_list, 'train_face': train_face_list, 'train_ldm': train_ldm_list,
                 'train_per': train_per_list, 'train_face_per': train_face_per_list,
                 'train_ldm_per': train_ldm_per_list,
                 'val': val_list, 'val_face': val_face_list, 'val_ldm': val_ldm_list,
                 'val_per': val_per_list, 'val_face_per': val_face_per_list, 'val_ldm_per': val_ldm_per_list,
                 'train_acc': train_acc_list, 'train_acc_per': train_acc_per_list, 'train_acc_neg': train_acc_neg_list,
                 'train_acc_neg_per': train_acc_neg_per_list, 'train_acc_pos': train_acc_pos_list,
                 'train_acc_pos_per': train_acc_pos_per_list,
                 'val_acc': val_acc_list, 'val_acc_per': val_acc_per_list, 'val_acc_neg': val_acc_neg_list,
                 'val_acc_neg_per': val_acc_neg_per_list, 'val_acc_pos': val_acc_pos_list,
                 'val_acc_pos_per': val_acc_pos_per_list,
                 'ldms_list': ldms_list,
                 'ldms_v_list': ldms_v_list
                 }
    anno_loss_list = pd.DataFrame(loss_list)
    anno_loss_list.to_csv(os.path.join(args.csv_dir, 'loss_train_%s_%s.csv' % (time_str, args.phase)))
    print('the train and val data is saved, path:{}'.format(
        os.path.join(args.csv_dir, 'loss_train_%s_%s.csv' % (time_str, args.phase))))


######predict landmarks
def predict(model, sample):
    model.eval()
    model.to('cpu')
    with torch.no_grad():
        inputs = sample['image']
        img = np.array(inputs.permute((1, 2, 0))).astype('uint8')
        landmarks = sample['landmarks']
        face = sample['face']
        inputs = inputs.unsqueeze(0)

        outputs_face, outputs_ldm = model(inputs)
        predict_face = torch.argmax(outputs_face, dim=1).item()
        if predict_face == 0:
            print('predict result:', predict_face)
            print('ground true:', sample['face'].item())
            print('face_sample is negative')
        else:
            print('predict result:', predict_face)
            print('ground true:', sample['face'].item())
            print('face_sample is postive')

        print('coordinate(MSE):{:.3f}'.format(np.mean((np.array(outputs_ldm.detach()).reshape(-1, ) -
                                                       np.array(landmarks.detach())) ** 2)))

        print('outputs:', np.array(outputs_ldm.detach()).reshape(-1, ))
        print('landmarks:', np.array(landmarks.detach()))

        landmarks = landmarks.reshape(-1, 2)
        outputs_ldm = outputs_ldm.reshape(-1, 2)
        fig1 = plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(img)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='.', color='r', label='org')
        plt.scatter(outputs_ldm[:, 0], outputs_ldm[:, 1], marker='.', color='g', label='pre')
        plt.legend()
        plt.subplot(122)
        img_2 = np.ones(img.shape)
        img_2[0, 0] = 0.0
        plt.imshow(img_2, cmap='gray')
        plt.scatter(landmarks[:, 0], landmarks[:, 1], marker='.', color='r', label='org')
        plt.scatter(outputs_ldm[:, 0], outputs_ldm[:, 1], marker='.', color='g', label='pre')
        plt.legend()
        # plt.show()
        fig2 = plt.figure(figsize=(6, 6))
        img = sample['img_name']
        rect = sample['rect']
        rect[2] = rect[2] - rect[0]
        rect[3] = rect[3] - rect[1]
        img = Image.open(img)
        _, h, w = transforms.ToTensor()(img).size()

        landmarks = landmarks * torch.tensor([rect[2] / train_boarder * 1.0, rect[3] / train_boarder * 1.0])
        landmarks += torch.tensor([rect[0], rect[1]])
        outputs_ldm = outputs_ldm * torch.tensor([rect[2] / train_boarder * 1.0, rect[3] / train_boarder * 1.0])
        outputs_ldm += torch.tensor([rect[0], rect[1]])
        plt.imshow(img)
        plt.gca().add_patch(plt.Rectangle((rect[0], rect[1]), rect[2], rect[3], color='b', fill=False, linewidth=1))
        if face == 1:
            plt.gca().scatter(outputs_ldm[:, 0], outputs_ldm[:, 1], marker='.', color='g', label='pre')
        plt.legend()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='trained_models',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',
                        help='training, predicting or finetuning')
    parser.add_argument('--csv-dir', type=str, default='csv',
                        help='csv directory')
    parser.add_argument('--face-rate', type=float, default=0.5,
                        help='postive and negative sample ratio')
    parser.add_argument('--loss-rate', type=float, default=0.5,
                        help='train face and landmarks ratio')
    parser.add_argument('--rorat-rate', type=float, default=0.0,
                        help='roratation rate')
    parser.add_argument('--rorat-anglemin', type=int, default=0,
                        help='roratation min angle')
    parser.add_argument('--rorat-anglemax', type=int, default=0,
                        help='roratation max angle')
    parser.add_argument('--arate', type=float, default=0.0,
                        help='affine rate')
    parser.add_argument('--aanglexmin', type=int, default=0,
                        help='affine x axis min angle')
    parser.add_argument('--aanglexmax', type=int, default=0,
                        help='affine x axis max angle')
    parser.add_argument('--aangleymin', type=int, default=0,
                        help='affine y axis min angle')
    parser.add_argument('--aangleymax', type=int, default=0,
                        help='affine y axis max angle')
    parser.add_argument('--sample-id', type=int, default=0,
                        help='sample id')

    args = parser.parse_args()

    path = './'
    # path = 'E:\AI\开课吧\人工智能第五期cv深度学习与计算机视觉班\lesson07\项目三'
    os.chdir(path)
    dir = os.getcwd()
    args.csv_dir = os.path.join(dir, args.csv_dir)
    args.save_directory = os.path.join(dir, args.save_directory)
    args.phase = 'predict'

    ##GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:1' if use_cuda else 'cpu')

    print('============>Loading Datasets')
    # print(os.getcwd())
    train_data = get_train_test_set('train3', args.rorat_rate, args.rorat_anglemin, args.rorat_anglemax, args.arate,
                                    args.aanglexmin, args.aanglexmax,
                                    args.aangleymin, args.aangleymax)
    test_data = get_train_test_set('val3')
    print('train samples:', len(train_data))
    print('val samples:', len(test_data))

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.test_batch_size)

    print('===========>Building Model')
    critierion_face = nn.CrossEntropyLoss(weight=torch.tensor([1 - args.face_rate, args.face_rate]))
    #     critierion_face = nn.CrossEntropyLoss()
    critierion_face = critierion_face.to(device)
    critierion_landmarks = nn.MSELoss()
    #     critierion_landmarks = nn.SmoothL1Loss()
    #    optimer = optim.Adam(model.parameters(), lr=args.lr)
    #     optimer = optim.SGD(model.parameters(), lr=args.lr)

    if args.phase == 'Train' or args.phase == 'train':
        print('training')
        model = Net().to(device)
        optimer = optim.Adam(model.parameters(), lr=args.lr)
        train(args, model, optimer, critierion_face, critierion_landmarks, train_loader, test_loader, device)
    elif args.phase == 'heatmap_train':
        model = Net2().to(device)
        print('heatmap training')
        optimer = optim.Adam(model.parameters(), lr=args.lr)
        train(args, model, optimer, critierion_face, critierion_landmarks, train_loader, test_loader, device, True)
    elif args.phase == 'finetune' or args.phase == 'Finetune':
        print('finetune')
        model = Net().to(device)
        model.load_state_dict(torch.load(os.path.join(args.save_directory,
                                                      'detector_20191218113625.pt')))

        optimer = optim.Adam(model.parameters(), lr=args.lr)
        train(args, model, optimer, critierion_face, critierion_landmarks, train_loader, test_loader, device)
    elif args.phase == 'train_res101':
        model_r = Net_Res()
        model_r = model_r.to(device)
        optimizer = optim.Adam(model_r.parameters(), lr=args.lr, betas=(0.9, 0.99))
        train(args, model_r, optimizer, critierion_face, critierion_landmarks, train_loader, test_loader, device)
    elif args.phase == 'finetune_res101':
        model_res101 = Net_Res()
        model_res101.load_state_dict(
            torch.load(os.path.join(args.save_directory, 'detector_20191223100930_train_res101.pt')))
        model_res101 = model_res101.to(device)
        optimizer = optim.Adam(model_res101.parameters(), lr=args.lr, betas=(0.9, 0.99))
        train(args, model_res101, optimizer, critierion_face, critierion_landmarks, train_loader, test_loader, device)
    elif args.phase == 'predict':
        print('predict')
        model = Net_Res18()
        model.load_state_dict(torch.load(os.path.join(args.save_directory, 'detector_20191222104232_finetune_res18.pt'),
                                         map_location=torch.device('cpu')))

        model.eval()
        sample = test_data[args.sample_id]
        print('sample.img_name:', sample['img_name'])
        predict(model, sample)
    elif args.phase == 'predict_res101':
        print('predict_res101')
        model = Net_Res()
        model.load_state_dict(torch.load(os.path.join(args.save_directory, 'detector_20191222104232_finetune_res18.pt'),
                                         map_location=torch.device('cpu')))

        model.eval()
        sample = test_data[7]
        print('sample.img_name:', sample['img_name'])
        predict(model, sample)


#         nums=len(test_data)
#         lists=predict_mse(model,test_data,nums,10,1000,True)
#         print(lists)


if __name__ == '__main__':
    main()
