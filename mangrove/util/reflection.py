# -*- coding: utf-8 -*-
import sys

__author__ = "ujihirokazuya"


def new_instance(module_name: str, class_name: str, args=None):
    __import__(module_name)
    clazz = getattr(sys.modules[module_name], class_name)
    if args:
        return clazz(args)
    return clazz()
