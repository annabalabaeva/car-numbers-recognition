# Author: Balabaeva Anna
# 28-02-2019

import torch
import numpy as np
import cv2
from random import random, randint, uniform


class ImageNormalization(object):
    def __call__(self, image):
        return image / np.float32(255)


class ImageNormalizationMeanStd(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


class ToTensor(object):
    def __call__(self, image):
        image = image.astype(np.float32)
        return torch.unsqueeze(torch.from_numpy(image), 0)


class ToType(object):
    def __call__(self, image):
        return image.astype(np.float32)


class Scale:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        # h, w = image.shape
        # if h>w:
        #     out_w, out_h = int(w/h*self.size), self.size
        # else:
        #     out_w, out_h = self.size, int(h/ w * self.size)
        # image_resized = cv2.resize(image, (out_w, out_h))
        # out_image = np.zeros((self.size, self.size))
        # shift_h = (self.size - out_h) // 2
        # shift_w = (self.size - out_w) // 2
        # out_image[shift_h:shift_h+out_h, shift_w:shift_w+out_w] = image_resized
        # return out_image
        h, w = image.shape
        h_ratio = h/self.size[0]
        w_ratio = w/self.size[1]
        if h_ratio > w_ratio:
            out_shape = (self.size[1], int(h/w*self.size[1]))
        else:
            out_shape = (int(w/h*self.size[0]), self.size[0])
        return cv2.resize(image, out_shape)


class CentralCrop(object):
    def __init__(self, out_size):
        self.out_height = out_size[0]
        self.out_width = out_size[1]

    def __call__(self, image):
        h, w = image.shape
        start_h = (h - self.out_height)//2
        start_w = (w - self.out_width)//2
        cropped_image = image[start_h:start_h+self.out_height, start_w:start_w+self.out_width]
        return cropped_image


class RandomCrop(object):
    def __init__(self, out_size):
        self.out_height = out_size[0]
        self.out_width = out_size[1]

    def __call__(self, image):
        pad_image = self._pad_if_needed(image)
        h, w = pad_image.shape
        start_h = randint(0, h - self.out_height)
        start_w = randint(0, w - self.out_width)
        cropped_image = self._crop(pad_image, start_h, start_w)
        return cropped_image

    def _pad_if_needed(self, image):
        img_height, img_width = image.shape[:2]
        if self.out_height > img_height:
            out_image = np.zeros((self.out_height, img_width))
            start = (self.out_height - img_height)//2
            out_image[start:start+img_height] = image
            img_height = self.out_height
        else:
            out_image = image
        if self.out_width > img_width:
            pad_image = np.zeros((img_height, self.out_width))
            start = (self.out_width - img_width)//2
            pad_image[:, start:start+img_width] = out_image
            out_image = pad_image
        return out_image

    def _crop(self, image, top, left):
        img_height, img_width = image.shape
        assert(top >= 0 and left >= 0)
        assert(top + self.out_height <= img_height)
        assert(left + self.out_width <= img_width)
        return image[top:top + self.out_height, left:left+self.out_width]


class RandomFlip(object):
    def __init__(self, p=0.5, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, image):
        if self.horizontal:
            if random() > self.p:
                image = image[:, ::-1]
        if self.vertical:
            if random() > self.p:
                image = image[::-1]
        return image


class RandomBrightness(object):
    def __init__(self, min_beta=-30, max_beta=70):
        self.max_beta = max_beta
        self.min_beta = min_beta

    def __call__(self, image):
        r = uniform(self.min_beta, self.max_beta)
        image_out = np.clip(image + r, 0, 255)
        return image_out


class RandomContrast(object):
    def __init__(self, min_alpha=0.7, max_alpha=1.3):
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha

    def __call__(self, image):
        r = uniform(self.min_alpha, self.max_alpha)
        image_out = np.uint8(np.clip(r*(np.int16(image)-127)+127, 0, 255))
        return image_out


def get_transforms(image_size):
    pass
    # transform = # USE COMPOSE TO APPLY ALL YOUR TRANSFORMS
    # return transform
