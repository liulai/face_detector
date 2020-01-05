import numpy as np
import cv2
import math
import random

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools
import matplotlib.pyplot as plt
import os
import torchvision.transforms.functional as F

from data_preprocess_pos_neg import show_image

from torchvision.transforms import transforms

# from data_myself import

###### negative sample landmarks:-1
# landmarks_neg = torch.zeros(42, dtype=torch.float)
landmarks_neg = np.zeros(42)
landmarks_neg_torch = torch.zeros(42, dtype=torch.float)

###image height and weight
train_boarder = 112
# train_boarder = 270

###heatmap image height and weight
# train_boarder_heatmap = 32
train_boarder_heatmap = 112


class Normalize(object):
    """
        Resieze to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        image_resize = np.asarray(
            image.resize((train_boarder, train_boarder), Image.BILINEAR),
            dtype=np.float32)  # Image.ANTIALIAS)
        image = channel_norm(image_resize)
        return {'image': image,
                'landmarks': landmarks
                }


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


class ToTensor(object):
    def __call__(self, sample):
        image, landmarks, landmarks_heatmap = sample['image'], sample['landmarks'], sample[
            'landmarks_heatmap']
        # image = np.expand_dims(image, axis=0).astype('float32')

        image = np.array(image).astype('float32')
        # print('image.size(after):',torch.from_numpy(image).permute(0,3,1,2).size())
        return {'image': torch.from_numpy(image.transpose((2, 0, 1))),
                'landmarks': torch.from_numpy(landmarks.astype('float32')),
                'landmarks_heatmap': landmarks_heatmap}


##train_boarder=112
class ToTensor1(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # count = sample['count']
        # image = np.expand_dims(image, axis=0).astype('float32')
        # image = transforms.RandomHorizontalFlip(0.5)(image)
        # image = transforms.RandomAffine(90)(image)
        image = np.array(image).astype('float32')
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        # image=transforms.RandomRotation(45)(image)
        return {'image': image,
                'landmarks': torch.from_numpy(landmarks.astype('float32'))}


##train_boarder=112, face and landmarks, 3 channels
class ToTensor2(object):
    def __call__(self, sample):
        image, landmarks, face = sample['image'], sample['landmarks'], sample['face']
        image = np.array(image).astype('float32')
        image = torch.from_numpy(image.transpose((2, 0, 1)))
        face = torch.from_numpy(face.astype('int64'))
        # print('face.size:',face)
        if int(face) == 1:
            # if landmarks != landmarks_neg:
            return {'image': image,
                    'landmarks': torch.from_numpy(landmarks.astype('float32')), 'face': face}
        else:
            return {'image': image,
                    'landmarks': torch.from_numpy(landmarks.astype('float32')), 'face': face}


##train_boarder=112, face and landmarks, 1 channels
class ToTensor3(object):
    def __call__(self, sample):
        image, landmarks, face = sample['image'], sample['landmarks'], sample['face']
        # image = np.array(image).astype('float32')
        # image = torch.from_numpy(image.transpose((2, 0, 1)))
        image = np.expand_dims(image, axis=0).astype('float32')
        # print('image.size:',image.shape)
        face = torch.from_numpy(face.astype('int64'))
        # print('face.size:',face)
        return {'image': image, 'landmarks': torch.from_numpy(landmarks.astype('float32')), 'face': face}
        # if landmarks is not None:
        #     return {'image': image,
        #             'landmarks': torch.from_numpy(landmarks.astype('float32')), 'face': face}
        # else:
        #     return {'image': image,
        #             'landmarks': landmarks_neg, 'face': face}


class ToRotation(object):
    '''
    turn Rotation
    '''

    def __init__(self, rate=0.0, degress=0):
        '''

        :param rate: change rate
        '''
        self.rate = rate
        self.degress = degress

    def __call__(self, sample):
        # print('rate:', self.rate)
        image, landmarks = sample['image'], sample['landmarks']

        if (np.random.choice(a=[0, 1], size=1, p=[1 - self.rate, self.rate]) == 1):
            image = F.rotate(image, self.degress, expand=True, center=(image.size[0] / 2, image.size[1] / 2))
            landmarks = landmarks.reshape(-1, 2) - (image.size[0] / 2, image.size[1] / 2)
            if (isinstance(self.degress, int)):
                angle = self.degress / 180.0 * np.pi
            else:
                angle = self.degress

            R = np.sqrt(landmarks[:, 0] ** 2 + landmarks[:, 1] ** 2)
            sita = np.arccos(landmarks[:, 0] / R)
            for i in range(len(landmarks)):
                if (landmarks[i, 1] >= 0):  ###One quadrant and Two quadrant
                    sita[i] = sita[i]
                elif (landmarks[i, 1] < 0):  ####Three quadrant and Four quadrant
                    sita[i] = 2 * np.pi - sita[i]
            landmarks_new = [np.cos(-angle + sita), np.sin(-angle + sita)] * R

            landmarks = landmarks_new.transpose((1, 0)) + (image.size[0] / 2, image.size[1] / 2)
            landmarks = landmarks.reshape(-1)
        return {'image': image, 'landmarks': landmarks}


class ToRotation2(object):
    '''
    turn Rotation
    '''

    def __init__(self, rate=0.0, anglemin=0, anglemax=0):
        '''

        :param rate: change rate
        :param angle: the angle changed is bigger than -angle, and less than angle
        '''
        if rate < 0.0 or rate > 1.0:
            print('the rate is less than 1.0, and bigger than 0.0')
            return
        self.rate = rate
#        self.angle = angle
        self.anglemin = anglemin
        self.anglemax = anglemax

    def __call__(self, sample):
        # print('rate:', self.rate)
        img, landmarks, face = sample['image'], sample['landmarks'], sample['face']

        if (np.random.choice(a=[0, 1], size=1, p=[1 - self.rate, self.rate]) == 1):
            img = np.array(img)
            img = np.array(img).astype('uint8')
            w, h, c = img.shape
#            angle1 = random.uniform(-self.angle, self.angle)
            angle1 = random.uniform(self.anglemin, self.anglemax)
            angle1 = random.choice([-angle1, angle1])


            angle2 = angle1 % 90
            M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle1,
                                        1 / (math.sqrt(2) * math.cos(math.fabs(math.pi / 4 - math.pi / 180 * angle2))))
            img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

            # print('face:',int(face))
            if int(face) == 1:
                landmarks = landmarks.reshape(-1, 2)
                landmarks = np.column_stack((landmarks, np.ones(len(landmarks))))
                landmarks = np.dot(np.array(landmarks), M.T)
                landmarks = landmarks.reshape(-1, )

        return {'image': img, 'landmarks': landmarks, 'face': face, 'hheight': sample['hheight'],
                'hweight': sample['hweight'],
                'sigma': sample['sigma']}


class ToAffine(object):
    '''
    turn Affine(front and back, or left and right)
    '''

    def __init__(self, rate=0.0, anglexmin=0, anglexmax=0, angleymin=0, angleymax=0, expand=30):
        '''

        :param rate: change rate
        :param anglex:
        :param
        '''
        if rate < 0.0 or rate > 1.0:
            print('the rate is less than 1.0, and bigger than 0.0')
            return
        self.rate = rate
#        self.anglex = anglex
#        self.angley = angley
        self.anglexmin = anglexmin
        self.anglexmax = anglexmax
        self.angleymin = angleymin
        self.angleymax = angleymax
        self.anglez = 0
        self.expand = expand

    def __call__(self, sample):
        # print('rate:', self.rate)
        img, landmarks, face = sample['image'], sample['landmarks'], sample['face']

        if (np.random.choice(a=[0, 1], size=1, p=[1 - self.rate, self.rate]) == 1):

#            anglex = random.uniform(-self.anglex, self.anglex)
#            angley = random.uniform(-self.angley, self.angley)

            anglex = random.uniform(self.anglexmin, self.anglexmax)
            anglex = random.choice([-anglex, anglex])

            angley = random.uniform(self.angleymin, self.angleymax)
            angley = random.choice([-angley, angley])

#            print('anglex:',anglex)
#            print('angley:',angley)
           # anglex = self.anglex
           # angley = self.angley

            img = np.array(img)
            img = np.array(img).astype('uint8')
            w, h, c = img.shape

            def rad(x):
                return x * np.pi / 180

            # 扩展图像，保证内容不超出可视范围
            # img = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, 0)
            img = cv2.copyMakeBorder(img, self.expand, self.expand, self.expand, self.expand, cv2.BORDER_CONSTANT, 0)
            w, h = img.shape[0:2]

            # anglex = 0
            # angley = 180
            anglez = 0  # 是旋转
            fov = 42
            # 镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
            z = np.sqrt(w ** 2 + h ** 2) / 2 / np.tan(rad(fov / 2))
            # 齐次变换矩阵
            rx = np.array([[1, 0, 0, 0],
                           [0, np.cos(rad(anglex)), -np.sin(rad(anglex)), 0],
                           [0, -np.sin(rad(anglex)), np.cos(rad(anglex)), 0, ],
                           [0, 0, 0, 1]], np.float32)

            ry = np.array([[np.cos(rad(angley)), 0, np.sin(rad(angley)), 0],
                           [0, 1, 0, 0],
                           [-np.sin(rad(angley)), 0, np.cos(rad(angley)), 0, ],
                           [0, 0, 0, 1]], np.float32)

            rz = np.array([[np.cos(rad(self.anglez)), np.sin(rad(self.anglez)), 0, 0],
                           [-np.sin(rad(self.anglez)), np.cos(rad(self.anglez)), 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], np.float32)

            r = rx.dot(ry).dot(rz)

            # 四对点的生成
            pcenter = np.array([h / 2, w / 2, 0, 0], np.float32)

            p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
            p2 = np.array([w, 0, 0, 0], np.float32) - pcenter
            p3 = np.array([0, h, 0, 0], np.float32) - pcenter
            p4 = np.array([w, h, 0, 0], np.float32) - pcenter

            list_dst = [r.dot(p1), r.dot(p2), r.dot(p3), r.dot(p4)]
            org = np.array([[0, 0],
                            [w, 0],
                            [0, h],
                            [w, h]], np.float32)

            dst = np.zeros((4, 2), np.float32)

            # 投影至成像平面
            for i in range(4):
                dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
                dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

            warpR = cv2.getPerspectiveTransform(org, dst)

            img = cv2.warpPerspective(img, warpR, (w, h), borderValue=(0, 0, 0))
            img = cv2.resize(img, (train_boarder, train_boarder))

            if int(face) == 1:
                # if landmarks != landmarks_neg:
                landmarks = landmarks.reshape(-1, 2)
                landmarks = landmarks + np.array([self.expand, self.expand])
                landmarks = np.column_stack((landmarks, np.ones(len(landmarks))))
                landmarks = np.dot(np.array(landmarks), warpR.T)
                landmarks = landmarks / landmarks[:, -1].reshape(-1, 1)
                landmarks = landmarks[:, :2]
                landmarks = landmarks * (train_boarder / np.array([w, h]))
                landmarks = landmarks.reshape(-1, )

        return {'image': img, 'landmarks': landmarks, 'face': face, 'hheight': sample['hheight'],
                'hweight': sample['hweight'],
                'sigma': sample['sigma']}


class ToFlip3(object):
    '''
    turn left and right
    '''

    def __init__(self, rate):
        '''

        :param rate: change rate
        '''
        self.rate = rate

    def __call__(self, sample):
        # print('rate:', self.rate)
        image, landmarks = sample['image'], sample['landmarks']
        # print('landmarks:',landmarks)
        # print('no flip')
        # count=0
        if (np.random.choice(a=[0, 1], size=1, p=[1 - self.rate, self.rate]) == 1):
            image = np.flip(np.array(image), 1)
            landmarks = np.abs(np.array([train_boarder * 1.0, 0.]) - landmarks.reshape(-1, 2))
            landmarks = landmarks.reshape(-1, )
            # print('filp 3')
            # count+=1
        # print('landmarks:', landmarks)
        return {'image': image, 'landmarks': landmarks}


class ToFlip4(object):
    '''
    turn up and down
    '''

    def __init__(self, rate):
        '''

        :param rate: change rate
        '''
        self.rate = rate

    def __call__(self, sample):
        # print('rate:', self.rate)
        image, landmarks = sample['image'], sample['landmarks']

        if (np.random.choice(a=[0, 1], size=1, p=[1 - self.rate, self.rate]) == 1):
            image = np.flip(np.array(image), 0)
            landmarks = np.abs(np.array([0., train_boarder * 1.0]) - landmarks.reshape(-1, 2))
            landmarks = landmarks.reshape(-1, )
        return {'image': image, 'landmarks': landmarks}


class ToHeatMap(object):
    def __call__(self, sample):
        image, landmarks, hheight, hweight, sigma = sample['image'], sample['landmarks'], sample['hheight'], sample[
            'hweight'], sample['sigma']
        # landmarks_heatmap = None
        # if landmarks is not None:
        landmarks_heatmap = heatmap_landmarks(train_boarder_heatmap, train_boarder_heatmap,
                                              landmarks.reshape(-1, 2) * (train_boarder_heatmap / train_boarder),
                                              hheight, hweight, sigma)
        return {'image': image, 'landmarks': landmarks,
                'landmarks_heatmap': torch.from_numpy(landmarks_heatmap.astype('float32')),
                'hheight': hheight, 'hweight': hweight, 'sigma': sigma}


class ToFlip1(object):
    '''
    turn left and right
    '''

    def __init__(self, rate):
        '''

        :param rate: change rate
        '''
        self.rate = rate

    def __call__(self, sample):
        # print('rate:', self.rate)
        # print('sample:', sample)
        image, landmarks, img_name, hheight, hweight, sigma = sample['image'], sample['landmarks'], sample[
            'img_name'], sample['hheight'], sample['hweight'], sample['sigma']

        if (np.random.choice(a=[0, 1], size=1, p=[1 - self.rate, self.rate]) == 1):
            image = np.flip(np.array(image), 1)
            landmarks = np.abs(np.array([train_boarder * 1.0, 0.]) - landmarks.reshape(-1, 2))
            landmarks = landmarks.reshape(-1, )

        return {'image': image, 'landmarks': landmarks, 'img_name': img_name,
                'hheight': hheight, 'hweight': hweight, 'sigma': sigma}


class ToFlip2(object):
    '''
    turn up and down
    '''

    def __init__(self, rate):
        '''

        :param rate: change rate
        '''
        self.rate = rate

    def __call__(self, sample):
        # print('rate:', self.rate)
        image, landmarks, img_name, hheight, hweight, sigma = sample['image'], sample['landmarks'], sample[
            'img_name'], sample['hheight'], sample['hweight'], sample['sigma']

        if (np.random.choice(a=[0, 1], size=1, p=[1 - self.rate, self.rate]) == 1):
            image = np.flip(np.array(image), 0)
            landmarks = np.abs(np.array([0., train_boarder * 1.0]) - landmarks.reshape(-1, 2))
            landmarks = landmarks.reshape(-1, )
        return {'image': image, 'landmarks': landmarks, 'img_name': img_name,
                'hheight': hheight, 'hweight': hweight, 'sigma': sigma}


def get_train_test_set(path,rate=0.0,anglemin=0,anglemax=0,arate=0.0,aanglexmin=0,aanglexmax=0,aangleymin=0,aangleymax=0):
    return load_data(path,rate,anglemin,anglemax,arate,aanglexmin,aanglexmax,aangleymin,aanglexmax)


def load_data(phase,rate=0.0,anglemin=0,anglemax=0,arate=0.0,aanglexmin=0,aanglexmax=0,aangleymin=0,aangleymax=0):
    data_file = phase + '.txt'
    with open(data_file) as f:
        lines = f.readlines()
    if phase == 'Train3' or phase == 'train3':
        print('phase:', phase)
        tsfm = transforms.Compose([ToRotation2(rate,anglemin,anglemax),ToAffine(arate,aanglexmin,aanglexmax,aangleymin,aangleymax,20),ToTensor2()])
#        tsfm = transforms.Compose([ToRotation2(rate,angle), ToAffine(arate,aanglex,aangley, 20), ToHeatMap(), ToTensor()])
    elif phase == 'val3' or phase == 'test3' or phase == 'Val3' or phase == 'Test3':
        print('phase:', phase)
        # tsfm = transforms.Compose([ToHeatMap(), ToTensor()])
#        tsfm = transforms.Compose([ToHeatMap(),ToTensor()])
        tsfm = transforms.Compose([ToTensor2()])
    else:
        print('phase:', phase)
        # tsfm = transforms.Compose([ToFlip1(0),ToFlip2(0), ToHeatMap(), ToTensor()])
        # tsfm = transforms.Compose([ToTensor1()])
        tsfm = transforms.Compose([ToTensor2()])
    # datasets = FaceLandmarksDataset(lines, phase, tsfm)
   # datasets = FaceLandmarksDataset2(lines, phase, tsfm)
    # datasets = FaceLandmarksDataset_HeatMap(lines, phase, 3, 3, 1.5, tsfm)
    datasets = FaceLandmarksDataset_HeatMap2(lines, phase, 5, 5, 1.5, tsfm)
    return datasets



class FaceLandmarksDataset(Dataset):
    def __init__(self, lines, phase, tsfm):
        self.lines = lines
        self.phase = phase
        self.transform = tsfm

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line_list = line.strip().split()

        img_name = line_list[0]
        rect = np.array(line_list[1:5]).astype('float')
        rect[2] += rect[0]
        rect[3] += rect[1]
        landmarks = np.array(line_list[5:]).astype('float32').reshape(-1, 2)
        # img = Image.open(img_name).convert('L')
        img = Image.open(img_name).convert('RGB')
        img_crop = img.crop(tuple(rect))
        landmarks = landmarks * (train_boarder / np.array(img_crop.size))
        landmarks = landmarks.reshape(-1, )
        img_crop = img_crop.resize((train_boarder, train_boarder))
        sample = {'image': img_crop, 'landmarks': landmarks}
        sample = self.transform(sample)

        sample['img_name'] = img_name
        sample['rect'] = rect
        return sample


###heatmap landmarks
class FaceLandmarksDataset_HeatMap(Dataset):
    def __init__(self, lines, phase, hheight, hweight, sigma, tsfm):
        self.lines = lines
        self.phase = phase
        self.hheight = hheight
        self.hweight = hweight
        self.sigma = sigma
        self.transform = tsfm

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line_list = line.strip().split()

        img_name = line_list[0]
        rect = np.array(line_list[1:5]).astype('float')
        rect[2] += rect[0]
        rect[3] += rect[1]
        landmarks = np.array(line_list[5:]).astype('float32').reshape(-1, 2)
        img = Image.open(img_name).convert('RGB')
        img_crop = img.crop(tuple(rect))
        landmarks = landmarks * (train_boarder / np.array(img_crop.size))
        landmarks = landmarks.reshape(-1, )
        img_crop = img_crop.resize((train_boarder, train_boarder), Image.BILINEAR)
        sample = {'image': img_crop, 'landmarks': landmarks,
                  'img_name': img_name, 'hheight': self.hheight, 'hweight': self.hweight, 'sigma': self.sigma}
        sample = self.transform(sample)
        sample['img_name'] = img_name
        sample['rect'] = rect
        return sample


#######face and landmarks
class FaceLandmarksDataset2(Dataset):
    def __init__(self, lines, phase, tsfm):
        self.lines = lines
        self.phase = phase
        self.transform = tsfm

    def __len__(self):
        # print(self.lines[2])
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line_list = line.strip().split()

        img_name = line_list[0]
        rect = np.array(line_list[1:5]).astype('float')

        ####postive sample
        # landmarks=None
        landmarks = landmarks_neg
        if (len(line_list) > 6):
            landmarks = np.array(line_list[5:-1]).astype('float32').reshape(-1, 2)
            # landmarks = landmarks * (train_boarder / np.array(img_crop.size))
            landmarks = landmarks * (train_boarder / np.array([rect[2], rect[3]]))
            landmarks = landmarks.reshape(-1, )

        face = np.array(line_list[-1])

        rect[2] += rect[0]
        rect[3] += rect[1]


        # img = Image.open(img_name).convert('L')
        img = Image.open(img_name).convert('RGB')
        img_crop = img.crop(tuple(rect))

        img_crop = img_crop.resize((train_boarder, train_boarder), Image.ANTIALIAS)
        sample = {'image': img_crop, 'landmarks': landmarks, 'face': face}
        sample = self.transform(sample)
        sample['img_name'] = img_name
        sample['rect'] = rect
        return sample


###heatmap landmarks
class FaceLandmarksDataset_HeatMap2(Dataset):
    def __init__(self, lines, phase, hheight, hweight, sigma, tsfm):
        self.lines = lines
        self.phase = phase
        self.hheight = hheight
        self.hweight = hweight
        self.sigma = sigma
        self.transform = tsfm

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line_list = line.strip().split()

        img_name = line_list[0]
        rect = np.array(line_list[1:5]).astype('float')

        ####postive sample
        # landmarks=None
        landmarks = landmarks_neg
        if (len(line_list) > 6):
            landmarks = np.array(line_list[5:-1]).astype('float32').reshape(-1, 2)
            # landmarks = landmarks * (train_boarder / np.array(img_crop.size))
            landmarks = landmarks * (train_boarder / np.array([rect[2], rect[3]]))
            landmarks = landmarks.reshape(-1, )

        face = np.array(line_list[-1])

        rect[2] += rect[0]
        rect[3] += rect[1]
        # landmarks = np.array(line_list[5:]).astype('float32').reshape(-1, 2)
        img = Image.open(img_name).convert('RGB')
        img_crop = img.crop(tuple(rect))
        img_crop = img_crop.resize((train_boarder, train_boarder), Image.BILINEAR)

        sample = {'image': img_crop, 'landmarks': landmarks, 'face': face,
                  'img_name': img_name, 'hheight': self.hheight, 'hweight': self.hweight, 'sigma': self.sigma}
        sample = self.transform(sample)
        sample['img_name'] = img_name
        sample['rect'] = rect
        sample['face'] = torch.from_numpy(face.astype('int64'))
        return sample


# Compute gaussian kernel
def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, variance):
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(img_width):
        for y_p in range(img_height):
            dist_sq = (x_p - c_x) * (x_p - c_x) + \
                      (y_p - c_y) * (y_p - c_y)
            exponent = dist_sq / 2.0 / variance / variance
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map


def show_heatmap_landmarks(img, landmarks, landmarks_heatmap, hheight, hweight, sigma):
    '''

    :param img: image(tensor),c x h x w
    :param landmarks: coordinate x, y
    :param hheight: heatmap height
    :param hweight: heatmap weight
    :param sigma: heatmap gaussian sigma
    :return: none
    '''
    plt.subplot(131)
    plt.title('original image')

    plt.imshow(np.array(img).transpose((1, 2, 0)).astype('uint8'))
    plt.gca().scatter(landmarks[:, 0], landmarks[:, 1], marker='.', color='r')

    plt.subplot(132)
    plt.title('heatmap(original)')
    landmarks = np.array(landmarks).astype('int')
    landmarks[landmarks < 0] = 0

    c, height, weight = img.shape
    img_heatmap = heatmap_landmarks(height, weight, landmarks, hheight, hweight, sigma)
    img_heatmap = np.sum(img_heatmap, axis=0)
    plt.imshow(img_heatmap, cmap='gray')

    plt.subplot(133)
    plt.title('heatmap(outputs)')
    plt.imshow(torch.sum(landmarks_heatmap, 0), cmap='gray')
    plt.show()


def heatmap_landmarks(height, weight, landmarks, hheight, hweight, sigma):
    '''
    :param img: image(tensor),c x h x w
    :param landmarks: coordinate x, y
    :param hheight: heatmap height
    :param hweight: heatmap weight
    :param sigma: heatmap gaussian sigma
    :return: none
    '''

    # _, img_height, img_weight = img.shape

    img_height, img_weight = height, weight


    heatmap = HeatMap(hheight, hweight, sigma) * 255
    landmarks = np.array(landmarks).astype('int')
    landmarks[landmarks < 0] = 0
    landmarks_heatmap = np.zeros((len(landmarks), img_height, img_weight))

    for item in range(len(landmarks)):
        img_heatmap = np.zeros((img_height, img_weight))

        x_c, y_c = landmarks[item][0], landmarks[item][1]
        if (x_c >= weight or y_c >= height):
            continue
        else:
            x_1 = x_c - int(hweight / 2)
            x_2 = x_c + int(hweight / 2)
            y_1 = y_c - int(hheight / 2)
            y_2 = y_c + int(hheight / 2)
            x_1 = 0 if x_1 < 0 else x_1
            x_2 = img_weight - 1 if x_2 >= img_weight else x_2
            y_1 = 0 if y_1 < 0 else y_1
            y_2 = img_height - 1 if y_2 >= img_height else y_2
            img_heatmap[y_1:y_2 + 1, x_1:x_2 + 1] = heatmap[
                                                    y_1 - y_c + int(hheight / 2):y_2 - y_c + int(hheight / 2) + 1,
                                                    x_1 - x_c + int(hweight / 2):x_2 - x_c + int(hweight / 2) + 1]
            landmarks_heatmap[item] = img_heatmap
    return landmarks_heatmap


# compute heatmap
def HeatMap(height, weight, sigma):
    heatmap = np.zeros((height, weight))
    y_c = (height + 1) / 2.0 - 1
    x_c = (weight + 1) / 2.0 - 1
    for i in range(height):
        for j in range(weight):
            heatmap[i][j] = (i - y_c) ** 2 + (j - x_c) ** 2
    heatmap = -heatmap / 2 / sigma ** 2
    return np.exp(heatmap)


###show landmarks
def img_landmarks(img, landmarks, face):
    plt.imshow(np.array(img).astype('uint8').transpose((1, 2, 0)))
    if int(face) == 1:
        # if landmarks != landmarks_neg:
        landmarks = landmarks.reshape(-1, 2)
        plt.scatter(landmarks[:, 0], landmarks[:, 1], c='b')
    plt.show()


def show_rotation(sample, angle=0):
    img, landmarks = sample['image'], sample['landmarks']
    img = img.permute(1, 2, 0)
    img = np.array(img).astype('uint8')
    w, h, c = img.shape
    # angle = 30
    angle2 = angle % 90
    print('math.fabs:', math.fabs(-0.2))
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle,
                                1 / (math.sqrt(2) * math.cos(math.fabs(math.pi / 4 - math.pi / 180 * angle2))))
    img = cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))
    landmarks = landmarks.reshape(-1, 2)
    landmarks = torch.cat((landmarks, torch.ones((landmarks.size(0), 1))), dim=1)
    print(torch.from_numpy(M.T.astype('float32')).size())
    landmarks = np.dot(np.array(landmarks), M.T)
    plt.imshow(img)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='b')
    plt.show()


def main():
    # print('main()')
    
    sample = train_data[3]
    


if __name__ == '__main__':
    main()
