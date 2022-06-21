# -*- coding: utf-8 -*-
"""
This file contains the PyTorch dataset for hyperspectral images and
related helpers.
"""
import spectral
import numpy as np
import torch
import torch.utils
import os
from tqdm import tqdm

try:
    # Python 3
    from urllib.request import urlretrieve
except ImportError:
    # Python 2
    from urllib import urlretrieve

from datasets.utils import open_file
import hdf5storage
#sample_gt
DATASETS_CONFIG_HSI = {
    'abu-airport-2': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat'
    },

    'Sandiego': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat'
    },
    'SZ': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat'
    },

    'PaviaC': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat'],
        'img': 'Pavia.mat',
        'gt': 'Pavia_gt.mat'
    },
    'PaviaU': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'],
        'img': 'PaviaU.mat',
        'gt': 'PaviaU_gt.mat'
    },
    'KSC': {
        'urls': ['http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                 'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat'],
        'img': 'KSC.mat',
        'gt': 'KSC_gt.mat'
    },
    'IndianPines': {
        'urls': ['http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'],
        'img': 'Indian_pines_corrected.mat',
        'gt': 'Indian_pines_gt.mat'
    },
    'Botswana': {
        'urls': ['http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat'],
        'img': 'Botswana.mat',
        'gt': 'Botswana_gt.mat',
    }
}

try:
    from datasets.custom_datasets import CUSTOM_DATASETS_CONFIG

    DATASETS_CONFIG_HSI.update(CUSTOM_DATASETS_CONFIG)
except ImportError:
    pass


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


def get_dataset_HSI(dataset_name, target_folder="./", datasets=DATASETS_CONFIG_HSI):
    """ Gets the dataset specified by name and return the related components.
    Args:
        dataset_name: string with the name of the dataset
        target_folder (optional): folder to store the datasets, defaults to ./
        datasets (optional): dataset configuration dictionary, defaults to prebuilt one
    Returns:
        img: 3D hyperspectral image (WxHxB)
        gt: 2D int array of labels
        label_values: list of class names
        ignored_labels: list of int classes to ignore
        rgb_bands: int tuple that correspond to red, green and blue bands
    """
    palette = None

    if dataset_name not in datasets.keys():
        raise ValueError("{} dataset is unknown.".format(dataset_name))

    dataset = datasets[dataset_name]

    folder = target_folder + datasets[dataset_name].get('folder', dataset_name + '/')

    if dataset_name == 'PaviaC':
        # Load the image
        img = open_file(folder + 'Pavia.mat')['pavia']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'Pavia_gt.mat')['pavia_gt']

        label_values = ["Undefined", "Water", "Trees", "Asphalt",
                        "Self-Blocking Bricks", "Bitumen", "Tiles", "Shadows",
                        "Meadows", "Bare Soil"]

        ignored_labels = [0]
    elif dataset_name == 'abu-airport-2':
        mat = hdf5storage.loadmat(folder + 'abu-airport-2.mat')
        # img = open_file(folder + 'abu-airport-2.mat')['hsi']
        img = mat['hsi']
        gt = mat['hsi_gt']



        rgb_bands = (55, 41, 12)

        # gt = open_file(folder + 'abu-airport-2.mat')['hsi_gt']

        label_values = ["background", "anomaly"]

        ignored_labels = [2]
    elif dataset_name == 'Sandiego':
        # Load the image

        mat = hdf5storage.loadmat(folder + 'Sandiego.mat')
        # img = open_file(folder + 'abu-airport-2.mat')['hsi']
        img = mat['hsi']
        gt = mat['hsi_gt']

        rgb_bands = (55, 41, 12)

        # gt = open_file(folder + 'abu-airport-2.mat')['hsi_gt']

        label_values = ["background", "anomaly"]

        ignored_labels = [2]
    elif dataset_name == 'SZ':
        # Load the image
        mat = hdf5storage.loadmat(folder + 'SZ.mat')
        # img = open_file(folder + 'abu-airport-2.mat')['hsi']
        img = mat['hsi']
        gt = mat['hsi_gt']

        rgb_bands = (55, 41, 12)

        # gt = open_file(folder + 'abu-airport-2.mat')['hsi_gt']

        label_values = ["background", "anomaly"]

        ignored_labels = [2]
    elif dataset_name == 'PaviaU':
        # Load the image
        img = open_file(folder + 'PaviaU.mat')['paviaU']

        rgb_bands = (55, 41, 12)

        gt = open_file(folder + 'PaviaU_gt.mat')['paviaU_gt']

        label_values = ['Undefined', 'Asphalt', 'Meadows', 'Gravel', 'Trees',
                        'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']

        ignored_labels = [0]

    elif dataset_name == 'IndianPines':
        # Load the image
        img = open_file(folder + 'Indian_pines_corrected.mat')
        img = img['indian_pines_corrected']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'Indian_pines_gt.mat')['indian_pines_gt']
        label_values = ["Undefined", "Alfalfa", "Corn-notill", "Corn-mintill",
                        "Corn", "Grass-pasture", "Grass-trees",
                        "Grass-pasture-mowed", "Hay-windrowed", "Oats",
                        "Soybean-notill", "Soybean-mintill", "Soybean-clean",
                        "Wheat", "Woods", "Buildings-Grass-Trees-Drives",
                        "Stone-Steel-Towers"]

        ignored_labels = [0]

    elif dataset_name == 'Botswana':
        # Load the image
        img = open_file(folder + 'Botswana.mat')['Botswana']

        rgb_bands = (75, 33, 15)

        gt = open_file(folder + 'Botswana_gt.mat')['Botswana_gt']
        label_values = ["Undefined", "Water", "Hippo grass",
                        "Floodplain grasses 1", "Floodplain grasses 2",
                        "Reeds", "Riparian", "Firescar", "Island interior",
                        "Acacia woodlands", "Acacia shrublands",
                        "Acacia grasslands", "Short mopane", "Mixed mopane",
                        "Exposed soils"]

        ignored_labels = [0]

    elif dataset_name == 'KSC':
        # Load the image
        img = open_file(folder + 'KSC.mat')['KSC']

        rgb_bands = (43, 21, 11)  # AVIRIS sensor

        gt = open_file(folder + 'KSC_gt.mat')['KSC_gt']
        label_values = ["Undefined", "Scrub", "Willow swamp",
                        "Cabbage palm hammock", "Cabbage palm/oak hammock",
                        "Slash pine", "Oak/broadleaf hammock",
                        "Hardwood swamp", "Graminoid marsh", "Spartina marsh",
                        "Cattail marsh", "Salt marsh", "Mud flats", "Wate"]

        ignored_labels = [0]
    else:
        # Custom dataset
        img, gt, rgb_bands, ignored_labels, label_values, palette = ...
        CUSTOM_DATASETS_CONFIG[dataset_name]['loader'](folder)

    # Filter NaN out
    nan_mask = np.isnan(img.sum(axis=-1))
    if np.count_nonzero(nan_mask) > 0:
        print(
            "Warning: NaN have been found in the data. It is preferable to remove them beforehand. Learning on NaN data is disabled.")
    img[nan_mask] = 0
    gt[nan_mask] = 0
    ignored_labels.append(0)

    ignored_labels = list(set(ignored_labels))
    # Normalization
    img = np.asarray(img, dtype='float32')
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    return img, gt, label_values, ignored_labels, rgb_bands, palette


class HyperX_HSI(torch.utils.data.Dataset):
    """ Generic class for a hyperspectral scene """

    def __init__(self, data, gt, cfg):
        """
        Args:
        """
        super(HyperX_HSI, self).__init__()
        self.data = data
        self.label = gt
        self.name = cfg.settings['dataset_hsi']
        self.patch_size = cfg.settings['patch_size']
        self.ignored_labels = set(cfg.settings['ignored_labels'])
        self.flip_augmentation = cfg.settings['flip_augmentation']
        self.radiation_augmentation = cfg.settings['radiation_augmentation']
        self.mixture_augmentation = cfg.settings['mixture_augmentation']
        self.center_pixel = cfg.settings['center_pixel']
        supervision = cfg.settings['supervision']
        # Fully supervised : use all pixels with label not ignored
        if supervision == 'full':
            mask = np.ones_like(gt)
            for l in self.ignored_labels:
                mask[gt == l] = 0
        # Semi-supervised : use all pixels, except padding
        elif supervision == 'semi':
            mask = np.ones_like(gt)
        x_pos, y_pos = np.nonzero(mask)
        p = self.patch_size // 2
        self.data_pad = np.pad(self.data, ((p, p), (p,p),(0,0)), 'symmetric')
        self.label_pad = np.pad(self.label, ((p, p), (p, p)), 'symmetric')
        self.indices = np.array([(x, y) for x, y in zip(x_pos, y_pos)])
        self.labels = [self.label[x, y] for x, y in self.indices]
        np.random.shuffle(self.indices)

    @staticmethod
    def flip(*arrays):
        horizontal = np.random.random() > 0.5
        vertical = np.random.random() > 0.5
        if horizontal:
            arrays = [np.fliplr(arr) for arr in arrays]
        if vertical:
            arrays = [np.flipud(arr) for arr in arrays]
        return arrays

    @staticmethod
    def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1 / 25):
        alpha = np.random.uniform(*alpha_range)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        return alpha * data + beta * noise

    def mixture_noise(self, data, label, beta=1 / 25):
        alpha1, alpha2 = np.random.uniform(0.01, 1., size=2)
        noise = np.random.normal(loc=0., scale=1.0, size=data.shape)
        data2 = np.zeros_like(data)
        for idx, value in np.ndenumerate(label):
            if value not in self.ignored_labels:
                l_indices = np.nonzero(self.labels == value)[0]
                l_indice = np.random.choice(l_indices)
                assert (self.labels[l_indice] == value)
                x, y = self.indices[l_indice]
                data2[idx] = self.data[x, y]
        return (alpha1 * data + alpha2 * data2) / (alpha1 + alpha2) + beta * noise

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        p = self.patch_size // 2
        x, y = self.indices[i]
        x1, y1 = x, y
        x2, y2 = x1 + self.patch_size, y1 + self.patch_size


        data = self.data_pad[x1:x2, y1:y2]
        label = self.label_pad[x1:x2, y1:y2]


        if self.flip_augmentation and self.patch_size > 1:
            # Perform data augmentation (only on 2D patches)
            data, label = self.flip(data, label)
        if self.radiation_augmentation and np.random.random() < 0.1:
            data = self.radiation_noise(data)
        if self.mixture_augmentation and np.random.random() < 0.2:
            data = self.mixture_noise(data, label)

        # Copy the data into numpy arrays (PyTorch doesn't like numpy views)
        data = np.asarray(np.copy(data).transpose((2, 0, 1)), dtype='float32')
        label = np.asarray(np.copy(label), dtype='int64')

        # Load the data into PyTorch tensors
        data = torch.from_numpy(data)
        label = torch.from_numpy(label)
        # Extract the center label if needed
        if self.center_pixel and self.patch_size > 1:
            label = label[self.patch_size // 2, self.patch_size // 2]
        # Remove unused dimensions when we work with invidual spectrums
        elif self.patch_size == 1:
            data = data[:, 0, 0]
            label = label[0, 0]

        if label == 0:
            semi_label = label
        elif label == 1:
            semi_label = - label
        return data, label, semi_label, i, x, y
