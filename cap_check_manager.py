# -*- coding: utf-8 -*-

"""
@Author  : LiuBing
@Software: PyCharm
@File    : cap_check_manager.py
@Time    : 2020/7/23 15:55
@describe:  滑块验证码缺口检测管理类
"""
import math
import os
import random
from typing import Any

import cv2
import numpy
import torch
import numpy as np

from io import BytesIO

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from data.config import update_config
from modules.build_yolact import Yolact
from utils.augmentations import FastBaseTransform
from utils.cap_check.datasets import ImageFolder, ImageDataset
from utils.cap_check.models import Darknet
from utils.cap_check.utils import non_max_suppression, rescale_boxes
from utils.output_utils import NMS, after_nms
from utils.track_gen import GestureTrackGenerator


class CapCheckManager(object):
    """缺口检测管理类"""

    def __init__(self, model_def: str = "utils/cap_check/config/yolo3-tiny.cfg",
                 weights_path: str = "utils/cap_check/checkpoints/yolov3_ckpt.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # set up model
        self.model = Darknet(os.path.join(os.path.dirname(__file__), model_def), img_size=416).to(device)
        if weights_path.endswith(".weights"):
            self.model.load_darknet_weights(weights_path)
        else:
            self.model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), weights_path),
                                             map_location="cuda" if torch.cuda.is_available() else "cpu"))
        self.model.eval()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def cap_check(self, file_path: str = None, img_stream: BytesIO = None) -> dict:
        """
        缺口识别
        :param file_path: 图片路径（绝对路径）或基于当前文件的相对路径
        :param img_stream: 图片二进制流
        :return: 坐标点 {x, y, box_w, box_h}   备注: box_w, box_h 位置为整个矩阵 和宽 和高
        """
        if file_path:
            dataloader = DataLoader(
                ImageFolder(file_path, img_size=416),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        elif img_stream:
            dataloader = DataLoader(
                ImageDataset(img_stream, img_size=416),
                batch_size=1,
                shuffle=False,
                num_workers=0,
            )
        else:
            raise Exception("必须二选其一 [file_path|file_strem]")

        for batch_i, input_imgs in enumerate(dataloader):
            input_imgs = Variable(input_imgs.type(self.Tensor))

            with torch.no_grad():
                detections = self.model(input_imgs)
                detections = non_max_suppression(detections, 0.8, 0.4)

            if detections is not None:
                temp_img = np.array(img_stream)
                if detections[0] is not None:
                    detections = rescale_boxes(detections[0], 416, temp_img.shape[:2])
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                        box_w = x2 - x1
                        box_h = y2 - y1
                        return {"x": int(x1), "y": int(y1), "box_w": int(box_w), "box_h": int(box_h)}
            return {}


class GestureCheckManager(object):
    """手势检测管理类"""
    def __init__(self, weights_path: str = "utils/cap_check/checkpoints/my_5001.pth"):
        self.config = "res101_my_config"
        update_config(self.config)
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            cudnn.benchmark = True
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        self.net = Yolact()
        self.net.load_weights(weights_path, self.cuda)
        self.net.eval()
        if self.cuda:
            self.net = self.net.cuda()

    def cap_check(self, img_stream: BytesIO, width, height) -> Any:
        """
        缺口识别
        :params img_stream: 图片二进制流
        :params width: 图片宽度
        :params height: 图片高度
        :return: 坐标点 {x, y, box_w, box_h}   备注: box_w, box_h 位置为整个矩阵 和宽 和高
        """
        # pil 转 cv2
        img_origin = cv2.cvtColor(numpy.asarray(img_stream), cv2.COLOR_RGB2BGR)
        img_tensor = torch.from_numpy(img_origin).float()
        if self.cuda:
            img_tensor = img_tensor.cuda()
        img_h, img_w = img_tensor.shape[0], img_tensor.shape[1]
        img_trans = FastBaseTransform()(img_tensor.unsqueeze(0))
        with torch.no_grad():
            net_outs = self.net(img_trans)
            nms_outs = NMS(net_outs, False)
            results = after_nms(nms_outs, img_h, img_w, show_lincomb=False, crop_masks=not False,
                                visual_thre=0.3, img_name="test.jpg")
            class_ids, classes, boxes, masks = [x.cpu().numpy() for x in results]
            one_obj = np.tile(masks[0], (3, 1, 1)).transpose((1, 2, 0))
            one_obj = one_obj * img_origin
            one_obj = cv2.cvtColor(one_obj, cv2.COLOR_BGR2GRAY)  # 灰度
            one_obj = cv2.resize(one_obj, (width, height))
            _xy_list = []
            image = cv2.rotate(one_obj, cv2.ROTATE_90_CLOCKWISE)  # 压缩为一条线
            shape = image.shape
            last_idx = None
            # 遍历每一列
            for x in range(shape[0]):
                line = image[x]
                # 找到本列上所有的轨迹点
                idx_list = []
                for idx, v in enumerate(line):
                    if v > 0:
                        idx_list.append(idx)

                if len(idx_list) >= 2:
                    # 求中位数 # TODO
                    idx = int(np.median(idx_list))
                    # idx = int(np.mean([idx_list[0], idx_list[-1]]))
                elif len(idx_list) == 1:
                    idx = idx_list[0]
                else:
                    if not last_idx:
                        continue
                    idx = last_idx
                image[x] = np.zeros(line.shape)
                image[x][idx] = 255
                _xy_list.append((x, idx))

            xy_list = []
            # 纠正噪点
            last_y = None
            n = 8
            n_size = 255 * 2 * (n - 6)
            for x, y in _xy_list:
                # 求9邻域面积
                neighbor_area = image[
                                max(x - n, 0): min(x + n, shape[0] - 1) + 1,
                                max(y - n, 0): min(y + n, shape[1] - 1) + 1,
                                ]
                _size = neighbor_area.sum()
                if _size <= n_size:
                    image[x][y] = 0
                    if last_y:
                        image[x][last_y] = 255
                        y = last_y
                xy_list.append((x, shape[1] - y))
                last_y = y

            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            cv2.imwrite("move_data.jpg", image)
        return xy_list, self.calc_xy(xy_list, width, height)

    def find_end_point_frist(self, points):
        """
        寻找曲线拐点  仅寻找第一个
        :param points:
        :return:
        """
        points_length = len(points)
        # 判断方向
        flag = 0
        for i in range(1, points_length):
            flag = points[i][1] - points[0][1]
            if flag != 0:
                break
        if flag == 0:
            # 直线
            return [0, points_length - 1]
        flag = 1 if flag > 0 else -1
        c = 0
        # 寻找拐点
        for i in range(0, points_length - 1):
            if points[i + 1][1] * flag <= points[i][1] * flag:
                c = i
                break
        return [0, c]

    def find_end_points(self, points):
        """
        :param points:
        :return:
        """
        points = points.copy()
        # 寻找所有拐点
        end_points = []
        #
        while 1:
            start, end = self.find_end_point_frist(points)
            end_points.append(points[start])
            next_points = points[end + 1 :]
            if len(next_points) <= 1:
                end_points.append(points[end])
                if next_points:
                    end_points.append(next_points[-1])
                break
            points = next_points
        # 去掉密集拐点 密集小波动 导致移动轨迹采样太多
        new_end_points = [end_points[0]]
        last_x = end_points[0][0]
        for idx in range(1, len(end_points) - 1):
            if end_points[idx][0] - last_x < 4:
                continue
            new_end_points.append(end_points[idx])
            last_x = end_points[idx][0]
        end_points = new_end_points
        return end_points

    def calc_xy(self, xy_list, width, height):
        """
                计算移动轨迹
                    寻找所有拐点
                    拐点间曲线转换为直线
                    直线移动轨迹使用 TrackGenerator
                    拐点处加入随机停顿
                    组合所有直线轨迹
                :param xy_list:
                :params width: 图片宽度
                :params height: 图片高度
                :return:
                """
        # 寻找所有拐点
        end_points = self.find_end_points(xy_list.copy())

        # 计算每段移动轨迹并组合
        g = []
        target_points = []
        t_list = []
        for idx in range(len(end_points)):
            if idx == 0:
                continue
            start_point, end_point = end_points[idx - 1], end_points[idx]
            # 计算距离
            delta_x = end_point[0] - start_point[0]
            delta_y = end_point[1] - start_point[1]
            distince = int(math.sqrt(delta_x ** 2 + delta_y ** 2))

            tg = GestureTrackGenerator((distince,), (0, 2))
            tg_list = tg.get_track(0)
            t_max = max(t_list) if t_list else 0
            # 拐角停顿
            t_max += random.randint(0, 20)
            if delta_y != 0:
                # 以下将水平移动转换为斜方向移动
                # 计算三角函数值
                tan = abs(delta_y) / delta_x
                sin = tan / math.sqrt(1 + tan ** 2)
                cos = sin / tan
                for _, x, y, t in tg_list:
                    t = t + t_max
                    _x = x * cos
                    _y = x * sin
                    if delta_y < 0:
                        _y *= -1
                    _point = (int(start_point[0] + _x), int(start_point[1] + _y), t)
                    if _point[0] > end_point[0]:
                        _point = (end_point[0], end_point[1], t)
                    target_points.append(_point)
                    t_list.append(t)
            else:
                for _, x, y, t in tg_list:
                    t = t + t_max
                    _point = (int(start_point[0] + x), int(start_point[1] + 0), t)
                    if _point[0] > end_point[0]:
                        _point = (end_point[0], end_point[1], t)
                    target_points.append(_point)
                    t_list.append(t)
        #
        t = np.zeros((height, width))
        for _tg in target_points:
            x, y, ts = _tg
            g.append("{},{},{}".format(*_tg))
            if y >= 158:
                y = 157
            t[y, x] = 255
        return g


# 初始化
cap_check_manager = CapCheckManager(weights_path="utils/cap_check/checkpoints/yolov3_ckpt_54.pth")
gesture_check_manager = GestureCheckManager(weights_path="utils/cap_check/checkpoints/my_5001.pth")