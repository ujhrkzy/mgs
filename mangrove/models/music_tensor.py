# -*- coding: utf-8 -*-
from util.preconditions import check_not_null
import numpy as np
from numpy import ndarray

__author__ = "ujihirokazuya"
__date__ = "2017/05/14"


class MusicTensor(object):

    def __init__(self, tensor: ndarray, normalize_enable: bool):
        """
        音楽データ
        [Warning] 引数のtensorをそのまま引き継ぐ。
        :param tensor: 音源tensor [num examples, num time steps, block size]
        :param mean: tensorの平均値
        :param std: tensorのstandard deviation
        """
        check_not_null(tensor)
        self.tensor = tensor
        self.mean = None
        self.std = None
        if normalize_enable:
            self._normalize()

    def _normalize(self):
        mean_tensor = np.mean(np.mean(self.tensor, axis=0), axis=0)
        std_tensor = np.sqrt(np.mean(np.mean(np.abs(self.tensor - mean_tensor) ** 2, axis=0), axis=0))
        std_tensor = np.maximum(1.0e-8, std_tensor)
        self.tensor[:][:] -= mean_tensor
        self.tensor[:][:] /= std_tensor
        self.mean = mean_tensor
        self.std = std_tensor

    def normalize(self, other):
        """
        正規化する。
        [Warning] otherオブジェクトのmean, std をそのまま引き継ぐ。
        :param other: MusicTensor
        :return: None
        """
        self.tensor[:][:] -= other.mean
        self.tensor[:][:] /= other.std
        self.mean = other.mean
        self.std = other.std

    def denormalize(self, target):
        target *= self.std
        target += self.mean
        return target


class TrainingMusicTensor(object):

    def __init__(self, x: MusicTensor, y: MusicTensor):
        self._x = x
        self._y = y
