from __future__ import division

from .models import *
from .utils import *
from .datasets import *

from PIL import Image
from io import BytesIO

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable


def detect(file_path: str = None,
           image_stream: BytesIO = None,
           model_def: str = "config/yolov3-captcha.cfg",
           weights_path: str = "checkpoints/yolov3_ckpt.pth",
           img_size: int = 416) -> tuple:
    """
    缺口识别
    :param file_path: 图片路径（绝对路径）或基于当前文件的相对路径
    :param image_stream: 图片二进制流
    :param model_def: 模型定义文件的路径
    :param weights_path: 权重文件路径
    :param img_size: 图片大小
    :return: 坐标点 (x, y, w, h)   备注: w, h 位置为整个矩阵 和宽 和高
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set up model
    model = Darknet(os.path.join(os.path.dirname(__file__), model_def), img_size=img_size).to(device)
    if weights_path.endswith(".weights"):
        model.load_darknet_weights(weights_path)
    else:
        model.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), weights_path), map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    if file_path:
        dataloader = DataLoader(
            ImageFolder(file_path, img_size=img_size),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    elif image_stream:
        dataloader = DataLoader(
            ImageDataset(image_stream, img_size=img_size),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
    else:
        raise Exception("必须二选其一 [file_path|file_strem]")

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    for batch_i, input_imgs in enumerate(dataloader):
        input_imgs = Variable(input_imgs.type(Tensor))

        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, 0.8, 0.4)

        if detections is not None:
            temp_img = np.array(image_stream)
            detections = rescale_boxes(detections[0], img_size, temp_img.shape[:2])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_w = x2 - x1
                box_h = y2 - y1
                # print('bbox', (int(x1), int(y1), int(box_w), int(box_h)))
                return (int(x1), int(y1), int(box_w), int(box_h))
        return ()


if __name__ == '__main__':
    image = Image.open("test10.png")
    # detect(file_path="data/captcha/test")
    print(detect(image_stream=image))