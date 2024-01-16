# -*- coding: utf-8 -*-

"""
@Author  : LiuBing
@Software: PyCharm
@File    : track_gen.py
@Time    : 2020/9/21 17:28
@describe: 优化版本
"""
import random

import numpy as np


class CapVelocity(object):
    def __init__(self, offset=350):
        # TODO 需要根据总的时间长度进行一个拉伸
        if offset == 350:
            start_offset = offset
            end_offset = offset + 100
        else:
            start_offset = offset/np.random.randint(350, 450) * 10000
            end_offset = start_offset + 1000
        self.b = 1 / np.random.randint(start_offset, end_offset)
        self.init_v_left = random.randint(2, 10)
        self.init_v_right = random.randint(12, 15)
        self.y_step_count = 0
        self.init_velocity_y = [-0.05, 0.05, 0.1, 0.15, 0.2]
        self.init_velocity_y += (
            np.random.randint(0, 15, len(self.init_velocity_y)) / 1000
        )

    def velocity_x(self, t):
        if t == 0:
            v = np.random.randint(self.init_v_left, self.init_v_right) / 100
        else:
            v = np.random.randint(10, 50, 1) / 100 * np.exp(-self.b * t) + np.random.randint(20, 100, 1) / 500 * np.exp(-self.b * t / 100)
        return float(v)

    def velocity_y(self, num_point=10):
        """
        :param num_point: 代表在多少个时间点内必须出现 v!=0 的情况
        :return:
        """
        if num_point == 0:
            num_point = 2
        if self.y_step_count == 0:
            v = random.randint(-5, 20) / 1000
        elif self.y_step_count == 1:
            v = np.random.choice([0, random.randint(6, 24) / 100], p=[0.7, 0.3])
        else:
            curr_step = np.random.choice(
                self.init_velocity_y, p=[0.1, 0.65, 0.12, 0.07, 0.06]
            )
            prob = 1.0 / num_point
            v = np.random.choice([0, curr_step], p=[1 - prob, prob])
        self.y_step_count += 1
        return float(v)


class GestureVelocity(object):
    def __init__(self):
        # TODO 需要根据总的时间长度进行一个拉伸
        self.b = 1 / np.random.randint(50, 150)

        self.init_v_left = random.randint(2, 10)
        self.init_v_right = random.randint(12, 15)
        self.y_step_count = 0
        self.init_velocity_y = [-0.05, 0.05, 0.1, 0.15, 0.2]
        self.init_velocity_y += (
            np.random.randint(0, 15, len(self.init_velocity_y)) / 1000
        )

    def velocity_x(self, t):
        if t == 0:
            v = np.random.randint(self.init_v_left, self.init_v_right) / 100
        else:
            v = np.random.randint(10, 200, 1) / 100 * np.exp(
                -self.b * t
            ) + np.random.randint(20, 100, 1) / 100 * np.exp(-self.b * t / 100)
        return float(v)

    def velocity_y(self, num_point=10):
        """
        :param num_point: 代表在多少个时间点内必须出现 v!=0 的情况
        :return:
        """
        if num_point == 0:
            num_point = 2
        if self.y_step_count == 0:
            v = random.randint(-5, 20) / 1000
        elif self.y_step_count == 1:
            v = np.random.choice([0, random.randint(6, 24) / 100], p=[0.7, 0.3])
        else:
            curr_step = np.random.choice(
                self.init_velocity_y, p=[0.1, 0.65, 0.12, 0.07, 0.06]
            )
            prob = 1.0 / num_point
            v = np.random.choice([0, curr_step], p=[1 - prob, prob])
        self.y_step_count += 1
        return float(v)


class CapTrackGenerator(object):
    """
    针对于缺口识别优化版本
    轨迹生成器调用示例:
        track_gen = TrackGenerator()
        track_data = track_gen.get_track(t_shift=1000)
    """

    def get_track(self, t_shift=None, x_offset=None, x_start=None):
        """
        :param t_shift: 时间整体偏移量
        :param x_offset: 总距离
        :param x_start: 起始位置
        :return:
        """
        time_list, x_point = self.get_x_track(x_offset, x_start)

        y_point = self.get_y_track(time_list)
        if t_shift:
            time_list = (
                np.array(time_list) + np.ones(len(time_list), dtype=np.int) * t_shift
            )
            time_list = time_list.tolist()
        return list(
            zip(
                # list(np.zeros(len(time_list), dtype=np.int)),
                x_point,
                y_point,
                time_list,
            )
        )

    def get_x_track(self, x_offset, x_start=None):
        if not x_start:
            x_start = random.randint(20, 40)
        total_distance = x_offset - x_start
        x_go = x_start
        time_list, go_list = [0], [x_start]
        v_1 = CapVelocity(x_offset)
        step_count, t = 0, 0
        while x_go <= total_distance + x_start:
            if step_count == 0:
                v = v_1.velocity_x(0)
                time_step = np.random.normal(70, 80)
                # TODO 优化 当获得值为负数值 转为正数 方式时间系数为负
                if time_step < 0:
                    time_step = abs(time_step)
            else:
                v = v_1.velocity_x(t)
                time_step = np.random.randint(2, 10)
            x_go += int(v * time_step)
            # t += time_step
            t += (time_step + random.randint(5, 10))
            time_list.append(int(t))
            go_list.append(x_go)
            step_count += 1
        return [time_list, go_list]

    def get_y_track(self, time_list):
        y_start, total_distance = random.randint(14, 20), random.randint(3, 6)
        end_area_length = random.randint(2000, 4000) / 5000
        y_go = y_start
        v_1 = CapVelocity()
        last_t = 0
        go_list = []
        for index, t in enumerate(time_list):
            if t == 0:
                v = 0
            elif y_go >= total_distance + y_start or index >= int(
                len(time_list) * end_area_length
            ):
                # 当前所走步数超过一定值 or 当前时间处于旅程的后半段，则不再往前走
                v = 0
            else:
                num_going = random.randint(3, 5)
                num_point = int(len(time_list) * (1 - end_area_length) / num_going)
                v = v_1.velocity_y(num_point)
            y_go += int(v * (t - last_t))
            go_list.append(y_go)
            last_t = t
        return go_list

    def method_test(self, num_track):
        rest = []
        for _ in range(num_track):
            rest.append(self.get_x_track())
        return rest


class GestureTrackGenerator(object):
    """
    手势验证码用
    轨迹生成器调用示例:
        track_gen = TrackGenerator()
        track_data = track_gen.get_track(t_shift=1000)
    """

    def __init__(self, total_distance_x=(), total_distance_y=()):

        # 初始变量调整
        self.total_distance_x = total_distance_x  # x轴移动距离
        self.total_distance_y = total_distance_y  # y轴波动距离
        # self.total_distance_x = (270, 275)  #x轴移动距离
        # self.total_distance_y = (6, 14) # y轴波动距离
        pass

    def get_track(self, t_shift=None):
        """
        :param t_shift: 时间整体偏移量
        :return:
        """
        time_list, x_point = self.get_x_track()
        y_point = self.get_y_track(time_list)
        if t_shift:
            time_list = (
                np.array(time_list) + np.ones(len(time_list), dtype=np.int) * t_shift
            )
            time_list = time_list.tolist()
        return list(
            zip(
                list(np.zeros(len(time_list), dtype=np.int)),
                x_point,
                y_point,
                time_list,
            )
        )

    def get_x_track(self):
        if len(self.total_distance_x) == 1:
            x_start, total_distance = 0, self.total_distance_x[0]
        else:
            x_start, total_distance = (
                random.randint(0, 10),
                random.randint(*self.total_distance_x),
            )
        x_go = x_start
        time_list, go_list = [0], [x_start]
        v_1 = GestureVelocity()
        step_count, t = 0, 0
        while x_go <= total_distance + x_start:
            if step_count == 0:
                v = v_1.velocity_x(0)
                # time_step = np.random.normal(250, 60)
                time_step = np.random.randint(30, 50)
            else:
                v = v_1.velocity_x(t)
                time_step = np.random.randint(15, 19)
            x_go += int(v * time_step)
            t += time_step
            time_list.append(int(t))
            go_list.append(x_go)
            step_count += 1
        return [time_list, go_list]

    def get_y_track(self, time_list):
        y_start, total_distance = (
            random.randint(0, 10),
            random.randint(*self.total_distance_y),
        )
        end_area_length = random.randint(2000, 4000) / 5000
        y_go = y_start
        v_1 = GestureVelocity()
        last_t = 0
        go_list = []
        for index, t in enumerate(time_list):
            if t == 0:
                v = 0
            elif y_go >= total_distance + y_start or index >= int(
                len(time_list) * end_area_length
            ):
                # 当前所走步数超过一定值 or 当前时间处于旅程的后半段，则不再往前走
                v = 0
            else:
                num_going = random.randint(3, 5)
                num_point = int(len(time_list) * (1 - end_area_length) / num_going)
                v = v_1.velocity_y(num_point)
            y_go += int(v * (t - last_t))
            go_list.append(y_go)
            last_t = t
        return go_list

    def method_test(self, num_track):
        rest = []
        for _ in range(num_track):
            rest.append(self.get_x_track())
        return rest