# -*- coding: utf-8 -*-
__author__ = "ujihirokazuya"


def get_not_null_value(value: str):
    return value if value else ""


def is_empty(value: str) -> bool:
    if not value:
        return True
    return len(value) == 0
