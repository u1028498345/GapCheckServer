# -*- coding: utf-8 -*-

"""
@Author  : LiuBing
@Software: PyCharm
@File    : server.py
@Time    : 2020/7/23 15:52
@describe: 缺口识别服务
"""
import os
import io
import time
import base64
from enum import unique, Enum

import aiofiles
from hashlib import md5

from aiohttp import web
from PIL import Image

from cap_check_manager import cap_check_manager, gesture_check_manager
from utils.utils import convert
from utils.logger import get_logger

routes = web.RouteTableDef()

logger = get_logger(logger_name="server", toFile=True)

# 统计images图片名称存放在内存中 用于去重
try:
    images_names = os.listdir("./images")
except:
    images_names = []

@unique
class ErrorCode(Enum):
    SUCCESS = 0       # 成功
    GAP_ERROR = 1     # 获取缺口位置失败
    PARAMS_ERROR = 2  # 请求参数错误
    ERROR = 3         # 错误


def response(code, message, times, data=None):
    """通用返回"""
    response_json = {"status": code.value, "msg": message, "data": data, "times": times}
    logger.info(response_json)
    return web.json_response(response_json)


@routes.post("/gap_check")
async def GapCheck(request: web.Request):
    """
    图片缺口检测
    Args:
        request:

        # post:参数
        @ data: 图片base64加密数据
        @ type: 图片类型 0 缺口验证码识别（极验，腾讯缺口等） 1 手势验证码识别 （58描绘轨迹）
    Returns:

    """
    start = time.time()
    query = await request.json()
    if not query.get("data"):
        return response(code=ErrorCode.PARAMS_ERROR, message="未提供缺口图数据, 请提供。", times=time.time() - start)
    if query.get("type", None) is None:
        return response(code=ErrorCode.PARAMS_ERROR, message="未提供识别类型，0：缺口验证码，1：手势验证码", times=time.time() - start)
        # 需要判断类型 目前支持缺口和轨迹识别（缺口识别毫秒级，轨迹秒级）
    if query.get("type") == 0:
        # 对数据进行base64解密
        try:
            img_data = base64.b64decode(query.get("data"))
        except:  # 解密异常
            return response(code=ErrorCode.ERROR, message="图像base64解密异常，请查看是否加密正确", times=time.time() - start)
        position = cap_check_manager.cap_check(img_stream=Image.open(io.BytesIO(img_data)))
        if position:
            return response(code=ErrorCode.SUCCESS, message="获取缺口位置成功", times=time.time() - start, data={"position": position})
        else:
            return response(code=ErrorCode.GAP_ERROR, message="获取缺口位置失败", times=time.time() - start, data={"position": position})
    elif query.get("type") == 1:
        # 手势验证码
        if not query.get("size"):
            return response(code=ErrorCode.ERROR, message="请提供图片大小(某些情况下真实图片宽高和要识别图像的宽高大小不同),比如：正常大小 300x400，\
             生成轨迹图片的大小 180x240，所以需要提供 180x240",
                            times=time.time() - start)
        # 对数据进行base64解密
        try:
            img_data = base64.b64decode(query.get("data"))
        except:  # 解密异常
            return response(code=ErrorCode.ERROR, message="图像base64解密异常，请查看是否加密正确", times=time.time() - start)
        try:
            w, g = [int(x) for x in query.get("size").split("x")]
        except:
            return response(code=ErrorCode.ERROR, message="提供图像的大小异常，示例：180x190", times=time.time() - start)

        position, calc = gesture_check_manager.cap_check(img_stream=Image.open(io.BytesIO(img_data)), width=w, height=g)
        if position:
            return response(code=ErrorCode.SUCCESS, message="获取缺口位置成功", times=time.time() - start, data={"position": position, "calc": calc})
        else:
            return response(code=ErrorCode.GAP_ERROR, message="获取缺口位置失败", times=time.time() - start, data={"position": position, "calc": calc})


# 目前只支持 缺口上传 其他暂不支持
@routes.post("/put_gap")
async def put_cap(request: web.Request):
    """获取成功的缺口识别图以及位置信息用于进行继续训练提高精准度"""
    start = time.time()
    query = await request.json()
    if not query.get("data") or isinstance(query.get("data"), str):
        return response(code=ErrorCode.PARAMS_ERROR, message="未提供缺口图以及位置信息数据, 请提供。", times=time.time() - start, data='示例:{"data"\
        :{"data": "图片base64加密数据"}, "postion": "x|y|w_box|h_box"  # 位置信息 , "size": "w|h"  # 图片宽高}')
    data = query.get("data")
    if not data.get("position"):
        return response(code=ErrorCode.PARAMS_ERROR, message="未提供位置信息数据, 请提供。", times=time.time() - start)
    elif not data.get("data"):
        return response(code=ErrorCode.PARAMS_ERROR, message="未提供缺口图数据, 请提供。", times=time.time() - start)
    elif not data.get("size"):
        return response(code=ErrorCode.PARAMS_ERROR, message="未提供缺口图宽高数据, 请提供。", times=time.time() - start)

    # 数据保存
    if not os.path.exists("./images"):  # 不存在创建
        os.mkdir("./images")
    if not os.path.exists("./labels"):  # 标签存储位置
        os.mkdir("./labels")

    # 对数据进行base64解密
    try:
        img_data = base64.b64decode(data.get("data"))
    except:  # 解密异常
        return response(code=ErrorCode.ERROR, message="图像base64解密异常，请查看是否加密正确", times=time.time() - start)

    name = md5(data.get("data").encode()).hexdigest()
    # 判断是否存在如果不存在再存储减少io操作
    if f"{name}.png" not in images_names:
        # 异步存储
        async with aiofiles.open(f"./images/{name}.png", mode="wb") as f:
            await f.write(img_data)
        # 存储 labels
        async with aiofiles.open(f"./labels/{name}.txt", mode="w") as f:
            #归一化
            position = data.get("position")
            position = [int(x) for x in position.split("|")]
            position[1], position[2] = (position[0] + position[2]), position[1]
            position[3] = position[2] + position[3]
            size = [int(x) for x in data.get("size").split("|")]
            bb = convert(size, position)
            await f.write("0 " + " ".join([str(a) for a in bb]) + '\n')

        # 添加到缓存中
        images_names.append(f"{name}.png")
    return response(code=ErrorCode.SUCCESS, message="保存成功", times=time.time() - start)


# 测试接口
@routes.get("/test")
async def test(request: web.Request):
    image = Image.open("test10.png").resize((100, 50))
    start = time.time()
    position = cap_check_manager.cap_check(img_stream=image)
    return response(code=ErrorCode.SUCCESS, message="", times=time.time() - start, data={"postion": position})


# 测试接口
@routes.get("/test1")
async def test1(request: web.Request):
    image = Image.open("download.jpg").resize((280, 158))
    image.save("download.jpg")
    start = time.time()
    position, calc = gesture_check_manager.cap_check(img_stream=image, width=280, height=158)
    return response(code=ErrorCode.SUCCESS, message="", times=time.time() - start, data={"postion": position, "calc": calc})

app = web.Application()
app.add_routes(routes)

if __name__ == '__main__':
    web.run_app(app, port=8004)