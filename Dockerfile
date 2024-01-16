FROM ubuntu:18.04

COPY build_file/sources.list /etc/apt/sources.list
# 设置时区
RUN apt-get update -y && apt-get install -y tzdata
RUN echo "Asia/Shanghai" > /etc/timezone
RUN rm -rf /etc/localtime && ln -s /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
# 设置apt
RUN apt-get update --fix-missing\
    && apt-get install -y language-pack-zh-hans vim python3 python3-pip libopencv-dev python-opencv\
    && apt-get clean

RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen &&   locale-gen
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


# 开始编译自己项目的环境
COPY . /GapCheckServer

WORKDIR /GapCheckServer

ENV PYTHONPATH "${PYTHONPATH}:/GapCheckServer"

RUN pip3 install --upgrade pip && pip3 install -i https://pypi.douban.com/simple scikit-build==0.16.7
RUN pip3 install -i https://mirrors.aliyun.com/pypi/simple --no-cache-dir -r requirements.txt \
    && rm -rf ~/.cache/pip;


CMD ["gunicorn server:app -k aiohttp.worker.GunicornWebWorker -b 0.0.0.0:8004 -w 5 --threads 4 --max-requests 2000 --max-requests-jitter 2000"]

