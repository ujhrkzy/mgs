# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from config.app_config import AppConfig

__author__ = "ujihirokazuya"
__date__ = "2017/05/14"


class ApplicationContext(object):

    def __init__(self):
        self._container = dict()

    @property
    def config(self) -> AppConfig:
        return self._container.get(AppConfig)

    def get(self, clazz):
        return self._container.get(clazz)

    def add(self, clazz, value):
        self._container[clazz] = value


context = ApplicationContext()
