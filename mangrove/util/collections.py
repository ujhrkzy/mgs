# -*- coding: utf-8 -*-

__author__ = "ujihirokazuya"


def is_empty_collection(value) -> bool:
    if value is None:
        return True
    return len(value) == 0


def is_match_list(values: list, condition) -> bool:
    if is_empty_collection(values):
        return False
    for value in values:
        if condition(value):
            return True
    return False
