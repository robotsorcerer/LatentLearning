# Ripped off: https://github.com/anniesch/dvd/blob/main/train.py
__all__ = ["ComposeMix", "RandomCrop", "RandomRotation", \
            "Scale", "RandomHorizontalFlip", \
            "RandomReverseTime", "UnNormalize"]

import torch
import random
import numpy as np
import numbers
import collections
import random
import cv2

class ComposeMix(object):
    r"""Composes several transforms together. It takes a list of
    transformations, where each element of transform is a list with 2
    elements. First being the transform function itself, second being a string
    indicating whether it's an "img" or "vid" transform
    Args:
        transforms (List[Transform, "<type>"]): list of transforms to compose.
                                                <type> = "img" | "vid"
    Example:
        transforms.ComposeMix([
        RandomCropVideo(84),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   td=[0.229, 0.224, 0.225])
    ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for transform in self.transforms:
            this_img = []
            for idx, img in enumerate(imgs):
                this_img.append(transform(img))
            imgs = this_img
        return imgs

class RandomCrop(object):
    r"""Crop the given video frames at a random location. Crop location is the
    same for all the frames.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_method (cv2 constant): Method to be used for padding.
    """

    def __init__(self, size, padding=0, pad_method=cv2.BORDER_CONSTANT):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_method = pad_method

    def __call__(self, imgs):
        """
        Args:
            img (numpy.array): Video to be cropped.
        Returns:
            numpy.array: Cropped video.
        """
        th, tw = self.size
        h, w = imgs[0].shape[:2]
        x1 = np.random.randint(0, w - tw)
        y1 = 0 if th >= h else np.random.randint(0, h - th)
        temps = [np.nan for i in range(len(imgs))]
        for idx, img in enumerate(imgs):
            if self.padding > 0:
                img = cv2.copyMakeBorder(img, self.padding, self.padding,
                                         self.padding, self.padding,
                                         self.pad_method)
            # sample crop locations if not given
            # it is necessary to keep cropping same in a video
            img_crop = img[y1:y1 + th, x1:x1 + tw]
            temps[idx] = img_crop
        return np.asarray(temps)

class RandomRotation(object):
    """Rotate the given video frames randomly with a given degree.
    Args:
        degree (float): degrees used to rotate the video
    """

    def __init__(self, degree=10):
        self.degree = degree

    def __call__(self, imgs):
        """
            Args:
                imgs (numpy.array): Video to be rotated.
            Returns:
                numpy.array: Randomly rotated video.
        """
        h, w = imgs[0].shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), self.degree, 1)

        temps = [np.nan for i in range(len(imgs))]
        for idx, img in enumerate(imgs):
            temps[idx] = cv2.warpAffine(img, M, (w, h))

        return np.asarray(temps)

    def __repr__(self):
        return self.__class__.__name__ + '(degree={})'.format(self.self.degree)

class Scale(object):
    r"""Rescale the input image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_LINEAR``
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy.array): Image to be scaled.
        Returns:
            numpy.array: Rescaled image.
        """
        assert isinstance(self.size, tuple), 'size must be a tuple.'
        
        return cv2.resize(img, tuple(self.size))

class RandomHorizontalFlip(object):
    """Horizontally flip the given video frames randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.array): Video to be flipped.
        Returns:
            numpy.array: Randomly flipped video.
        """
        if random.random() < self.p:
            for idx, img in enumerate(imgs):
                imgs[idx] = cv2.flip(img, 1)
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class RandomReverseTime(object):
    """Reverse the given video frames in time randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, imgs):
        """
        Args:
            imgs (numpy.array): Video to be flipped.
        Returns:
            numpy.array: Randomly flipped video.
        """
        if random.random() < self.p:
            imgs = imgs[::-1]
        return imgs

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)
    
class UnNormalize(object):
    """Unnormalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel x std) + mean
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = np.array(mean).astype('float32')
        self.std = np.array(std).astype('float32')

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        if isinstance(tensor, torch.Tensor):
            self.mean = torch.FloatTensor(self.mean)
            self.std = torch.FloatTensor(self.std)

            if (self.std.dim() != tensor.dim() or
                    self.mean.dim() != tensor.dim()):
                for i in range(tensor.dim() - self.std.dim()):
                    self.std = self.std.unsqueeze(-1)
                    self.mean = self.mean.unsqueeze(-1)

            tensor = torch.add(torch.mul(tensor, self.std), self.mean)
        else:
            # Relying on Numpy broadcasting abilities
            tensor = tensor * self.std + self.mean
        return tensor