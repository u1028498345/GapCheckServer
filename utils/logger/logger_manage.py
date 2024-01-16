# -*- coding: utf-8 -*-
"""
@author: Liubing
@software: PyCharm
@file: logger_manage.py
@time: 2019-08-10 18:00
@describe:
"""
import re
import os
import logging
import datetime
import platform
import logging.handlers

from collections import defaultdict


try:
    import codecs
except Exception:
    codecs = None

func_count_dict = defaultdict(int)

FILECOLOR = "0"
STDOTCOLOR = "1"
FMT = "%(asctime)-15s %(threadName)s %(filename)s:%(lineno)d %(levelname)s %(message)s"


class FileColoredFormatter(logging.Formatter):
    """文件颜色格式化程序"""
    def __init__(self, fmt=FMT):
        logging.Formatter.__init__(self, fmt=fmt)

    def format(self, record):
        return logging.Formatter.format(self, record)


class StdotColoredFormatter(logging.Formatter):
    """stdot颜色格式化程序"""
    def __init__(self, fmt=FMT):
        logging.Formatter.__init__(self, fmt=fmt)
        self.pt = platform.system()

    def format(self, record):
        COLORS = {
            'Black': '0;30',  # 黑色
            'Red': '0;31',  # 红色
            'Green': '0;32',  # 绿色
            'Yellow': '0;33',  # 棕色
            'Blue': '0;34',  # 蓝色
            'Purple': '0;35',  # 紫色
            'Cyan': '0;36',  # 青色
            'Light_Gray': '0;37',  # 浅灰色

            'Dark_Gray': '1;30',  # 黑灰色
            'Light_Red': '1;31',  # 浅红色
            'Light_Green': '1;32',  # 浅绿色
            'Light_Yellow': '1;33',  # 黄色
            'Light_Blue': '1;34',  # 浅蓝色
            'Light_Purple': '1;35',  # 浅紫色
            'Light_Cyan': '1;36',  # 浅青色
            'White': '1;37',  # 白色
        }
        COLOR_SEQ = "\033[%sm"
        RESET_SEQ = "\033[0m"

        message = logging.Formatter.format(self, record)

        # 优化在windows平台显示问题
        if self.pt == "Windows":
            return message

        if record.levelno == logging.DEBUG:
            message = COLOR_SEQ % COLORS['Light_Cyan'] + message + RESET_SEQ
        elif record.levelno == logging.INFO:
            message = COLOR_SEQ % COLORS['Light_Gray'] + message + RESET_SEQ
            pass
        elif record.levelno == logging.WARNING:
            message = COLOR_SEQ % COLORS['Yellow'] + message + RESET_SEQ
        elif record.levelno == logging.ERROR:
            message = COLOR_SEQ % COLORS['Red'] + message + RESET_SEQ
        elif record.levelno == logging.CRITICAL:
            message = COLOR_SEQ % COLORS['Purple'] + message + RESET_SEQ
        return message


class ColorFactory(object):
    """格式化程序简单工厂类"""

    @classmethod
    def build(self, name):
        return {
            "0": FileColoredFormatter(),
            "1": StdotColoredFormatter()
        }.get(name, "0")


class MultiprocessHandler(logging.FileHandler):
    """支持多进程的TimedRotatingFileHandler"""

    def __init__(self, filename, when='H', backupCount=0, encoding=None, delay=False):
        """
        :param filename 日志文件名
        :param when 时间间隔的单位
        :param backupCount 保留文件个数
        :param delay 是否开启 OutSteam缓存 True 表示开启缓存，OutStream输出到缓存，待缓存区满后，刷新缓存区，并输出缓存数据到文件。
            False表示不缓存，OutStrea直接输出到文件"""

        # 文件名更改
        filename = filename.rsplit("/", 1)
        self.prefix = os.path.join(os.path.join(filename[0], "%Y%m%d"), filename[1])
        self.backupCount = backupCount
        self.when = when.upper()

        # 正则匹配 年-月-日
        self.extMath = r"^\d{4}-\d{2}-\d{2}"

        # S 每秒建立一个新文件
        # M 每分钟建立一个新文件
        # H 每天建立一个新文件
        # D 每天建立一个新文件
        self.when_dict = {
            'S': "%Y-%m-%d-%H-%M-%S",
            'M': "%Y-%m-%d-%H-%M",
            'H': "%Y%m%d_%H",
            'D': "%Y-%m-%d",
            'MIDNIGHT': "%Y-%m-%d"
        }

        # 日志文件日期后缀
        self.suffix = self.when_dict.get(self.when)
        if not self.suffix:
            raise ValueError(u"指定的日期间隔单位无效: %s" % self.when)
        # 拼接文件路径 格式化字符串
        self.filefmt = "%s_%s.log" % (self.prefix[:-4], self.suffix)
        # 使用当前时间，格式化文件格式化字符串
        self.filePath = datetime.datetime.now().strftime(self.filefmt)
        # 获得文件夹路径
        _dir = os.path.dirname(self.filePath)
        try:
            # 如果日志文件夹不存在，则创建文件夹
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        except Exception:
            print(u"创建文件夹失败")
            print(u"文件夹路径：" + self.filePath)
            pass

        if codecs is None:
            encoding = None

        logging.FileHandler.__init__(self, self.filePath, 'a+', encoding, delay)

    def shouldChangeFileToWrite(self):
        """
        更改日志写入目的写入文件
        :return True 表示已更改，False 表示未更改
        """
        # 以当前时间获得新日志文件路径
        _filePath = datetime.datetime.now().strftime(self.filefmt)
        # 新日志文件日期 不等于 旧日志文件日期，则表示 已经到了日志切分的时候
        #   更换日志写入目的为新日志文件。
        # 例如 按 天 （D）来切分日志
        #   当前新日志日期等于旧日志日期，则表示在同一天内，还不到日志切分的时候
        #   当前新日志日期不等于旧日志日期，则表示不在
        # 同一天内，进行日志切分，将日志内容写入新日志内。
        if _filePath != self.filePath:

            # 这里添加如果更改了日期 创建对应的日期文件夹
            temp_dir = os.path.dirname(_filePath)
            if not os.path.exists(temp_dir):
                os.mkdir(temp_dir)
            self.filePath = _filePath
            return True
        return False

    def doChangeFile(self):
        """输出信息到日志文件，并删除多于保留个数的所有日志文件"""
        # 日志文件的绝对路径
        self.baseFilename = os.path.abspath(self.filePath)
        # stream == OutStream
        # stream is not None 表示 OutStream中还有未输出完的缓存数据
        if self.stream:
            # flush close 都会刷新缓冲区，flush不会关闭stream，close则关闭stream
            # self.stream.flush()
            self.stream.close()
            # 关闭stream后必须重新设置stream为None，否则会造成对已关闭文件进行IO操作。
            self.stream = None
        # delay 为False 表示 不OutStream不缓存数据 直接输出
        #   所有，只需要关闭OutStream即可
        if not self.delay:
            # 这个地方如果关闭colse那么就会造成进程往已关闭的文件中写数据，从而造成IO错误
            # delay == False 表示的就是 不缓存直接写入磁盘
            # 我们需要重新在打开一次stream
            # self.stream.close()
            self.stream = self._open()
        # 删除多于保留个数的所有日志文件
        if self.backupCount > 0:
            print('删除日志')
            for s in self.getFilesToDelete():
                print(s)
                os.remove(s)

    def getFilesToDelete(self):
        """获得过期需要删除的日志文件"""
        # 分离出日志文件夹绝对路径
        # split返回一个元组（absFilePath,fileName)
        # 例如：split('I:\ScripPython\char4\mybook\util\logs\mylog.2017-03-19）
        # 返回（I:\ScripPython\char4\mybook\util\logs， mylog.2017-03-19）
        # _ 表示占位符，没什么实际意义，
        dirName, _ = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        # self.prefix 为日志文件名 列如：mylog.2017-03-19 中的 mylog
        # 加上 点号 . 方便获取点号后面的日期
        prefix = self.prefix + '.'
        plen = len(prefix)
        for fileName in fileNames:
            if fileName[:plen] == prefix:
                # 日期后缀 mylog.2017-03-19 中的 2017-03-19
                suffix = fileName[plen:]
                # 匹配符合规则的日志文件，添加到result列表中
                if re.compile(self.extMath).match(suffix):
                    result.append(os.path.join(dirName, fileName))
        result.sort()

        # 返回  待删除的日志文件
        #   多于 保留文件个数 backupCount的所有前面的日志文件。
        if len(result) < self.backupCount:
            result = []
        else:
            result = result[:len(result) - self.backupCount]
        return result

    def emit(self, record):
        """发送一个日志记录
        覆盖FileHandler中的emit方法，logging会自动调用此方法"""
        try:
            if self.shouldChangeFileToWrite():
                self.doChangeFile()
            logging.FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def get_logger(logger_name="aa", path="./logs", toFile=False):
    """
    初始化 短信服务的 logger 部分，无返回值，logger 的模式，可以随时get到
    :param logger_name: 日志名称
    :param path: 存储位置
    :param toFile: 是否存储到文件
    :return:
    """

    logger_name = logger_name
    # getLogger 为单例模式
    api_logger = logging.getLogger(logger_name)
    api_logger.setLevel(logging.DEBUG)

    # handler 存在的判定
    if not api_logger.handlers:
        # rotating file logger
        # 进程不安全导致
        if toFile:
            # 如果输出到文件再创建文件夹
            log_path = path
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            log_fn = "{0}/{1}.log".format(log_path, logger_name)

            file_handle = MultiprocessHandler(log_fn, "h", encoding='utf-8')
            file_handle.setFormatter(ColorFactory.build(FILECOLOR))
            api_logger.addHandler(file_handle)
        steam_handler = logging.StreamHandler()
        steam_handler.setFormatter(ColorFactory.build(STDOTCOLOR))
        api_logger.addHandler(steam_handler)

    return api_logger
