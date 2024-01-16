# -*- coding: utf-8 -*-

"""
@Author  : LiuBing
@Software: PyCharm
@File    : test.py
@Time    : 2020/7/24 16:49
@describe: 
"""
import base64
from io import BytesIO

import requests
from PIL import Image


def test():
    image = Image.open("test10.png")
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode()
    while True:
        ret = requests.post("http://127.0.0.1:8004/gap_check", json={"data": base64_str, "type": 0})
        print(ret.json())
        break


def test1():
    image = Image.open("1.jpg")
    img_buffer = BytesIO()
    image.save(img_buffer, format="JPEG")
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode()
    while True:
        ret = requests.post("http://127.0.0.1:8004/gap_check", json={"data": base64_str, "type": 1, "size": "380x358"})
        print(ret.json())
        break

if __name__ == '__main__':
    # th = []
    #
    # for i in range(100):
    #     th.append(threading.Thread(target=test))
    #
    # for i in th:
    #     i.start()
    #
    # for i in th:
    #     i.join()
    # test()
    test1()