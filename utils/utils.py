# -*- coding: utf-8 -*-

"""
@Author  : LiuBing
@Software: PyCharm
@File    : utils.py
@Time    : 2020/8/25 12:06
@describe:  工具类
"""


def convert(size, box):
    """归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h
