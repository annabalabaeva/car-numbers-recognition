# Author: Balabaeva Anna
# 28-02-2019

import torch
import numpy as np
import cv2
import random


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
        h, w = image.shape
        h_ratio = h / self.size[0]
        w_ratio = w / self.size[1]
        if h_ratio > w_ratio:
            out_shape = (self.size[1], int(h / w * self.size[1]))
        else:
            out_shape = (int(w / h * self.size[0]), self.size[0])
        return cv2.resize(image, out_shape)


class RandomScale:
    def __init__(self, size, max_k=1.2):
        self.size = size
        self.max_k = max_k

    def __call__(self, image):
        random_k = random.uniform(1.0, self.max_k)
        out_size = (int(self.size[0]*random_k), int(self.size[1]*random_k))
        h, w = image.shape
        h_ratio = h / out_size[0]
        w_ratio = w / out_size[1]
        if h_ratio > w_ratio:
            out_shape = (out_size[1], int(h / w * out_size[1]))
        else:
            out_shape = (int(w / h * out_size[0]), out_size[0])
        return cv2.resize(image, out_shape)


class RandomPad:
    def __init__(self, min_scale):
        self.min_scale = min_scale

    def __call__(self, image):
        h, w = image.shape
        scale = random.uniform(self.min_scale, 1.0)
        out_w, out_h = (int(w * scale), int(h*scale))
        image_resized = cv2.resize(image, (out_w, out_h))
        start_h = random.randint(0, h-out_h)
        start_w = random.randint(0, w-out_w)
        out_image = np.zeros((h, w))
        out_image[start_h:start_h+out_h, start_w:start_w+out_w] = image_resized
        out_image[0:start_h, :] = out_image[start_h, :]
        out_image[start_h+out_h:, :] = out_image[start_h+out_h-1, :]
        out_image[:, 0:start_w] = out_image[:, start_w][:, np.newaxis]
        out_image[:, start_w+out_w:] = out_image[:, start_w+out_w-1][:, np.newaxis]
        return out_image


class CentralCrop(object):
    def __init__(self, out_size):
        self.out_height = out_size[0]
        self.out_width = out_size[1]

    def __call__(self, image):
        h, w = image.shape
        start_h = (h - self.out_height) // 2
        start_w = (w - self.out_width) // 2
        cropped_image = image[start_h:start_h + self.out_height, start_w:start_w + self.out_width]
        # print(cropped_image.shape)
        return cropped_image


class RandomCrop(object):
    def __init__(self, out_size):
        self.out_height = out_size[0]
        self.out_width = out_size[1]

    def __call__(self, image):
        pad_image = self._pad_if_needed(image)
        h, w = pad_image.shape
        start_h = random.randint(0, h - self.out_height)
        start_w = random.randint(0, w - self.out_width)
        cropped_image = self._crop(pad_image, start_h, start_w)
        # print(cropped_image.shape)
        return cropped_image

    def _pad_if_needed(self, image):
        img_height, img_width = image.shape[:2]
        if self.out_height > img_height:
            out_image = np.zeros((self.out_height, img_width))
            start = (self.out_height - img_height) // 2
            out_image[start:start + img_height] = image
            img_height = self.out_height
        else:
            out_image = image
        if self.out_width > img_width:
            pad_image = np.zeros((img_height, self.out_width))
            start = (self.out_width - img_width) // 2
            pad_image[:, start:start + img_width] = out_image
            out_image = pad_image
        return out_image

    def _crop(self, image, top, left):
        img_height, img_width = image.shape
        assert (top >= 0 and left >= 0)
        assert (top + self.out_height <= img_height)
        assert (left + self.out_width <= img_width)
        return image[top:top + self.out_height, left:left + self.out_width]


class RandomFlip(object):
    def __init__(self, p=0.5, horizontal=True, vertical=False):
        self.horizontal = horizontal
        self.vertical = vertical
        self.p = p

    def __call__(self, image):
        if self.horizontal:
            if random.random() > self.p:
                image = image[:, ::-1]
        if self.vertical:
            if random.random() > self.p:
                image = image[::-1]
        return image


class RandomBrightness(object):
    def __init__(self, min_beta=-90, max_beta=70):
        self.max_beta = max_beta
        self.min_beta = min_beta

    def __call__(self, image):
        r = random.uniform(self.min_beta, self.max_beta)
        image_out = np.clip(image + r, 0, 255)
        return image_out


class RandomContrast(object):
    def __init__(self, min_alpha=0.4, max_alpha=1.3):
        self.max_alpha = max_alpha
        self.min_alpha = min_alpha

    def __call__(self, image):
        r = random.uniform(self.min_alpha, self.max_alpha)
        image_out = np.uint8(np.clip(r * (np.int16(image) - 127) + 127, 0, 255))
        return image_out


class RandomBlur(object):
    def __init__(self, p=0.4):
        self.p = p

    def __call__(self, image):
        r = random.uniform(0.0, 1.0)
        if r < self.p:
            image_out = np.uint8(cv2.blur(image, (3, 3)))
        else:
            image_out = image
        return image_out


class RandomRotation(object):
    def __init__(self, p=0.6, min_angle=-10, max_angle=10):
        self.max_angle = max_angle
        self.min_angle = min_angle
        self.p = p

    def __call__(self, image):
        r = random.uniform(0.0, 1.0)
        if r < self.p:
            angle = random.uniform(self.min_angle, self.max_angle)
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image_out = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_AREA,
                                       borderMode=cv2.BORDER_REPLICATE)
        else:
            image_out = image
        return image_out


def get_transforms(image_size):
    pass
    # transform = # USE COMPOSE TO APPLY ALL YOUR TRANSFORMS
    # return transform
