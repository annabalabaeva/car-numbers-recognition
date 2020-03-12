FROM debian:sid

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget \
    python3 python3-pip python3-dev python3-setuptools \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev libglib2.0-0 &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

ENV PYTHONPATH $PYTHONPATH:/workdir/src

RUN pip3 install --no-cache-dir numpy

RUN pip3 install --no-cache-dir torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install --no-cache-dir trafaret trafaret-config
RUN pip3 install --no-cache-dir deep-pipe
RUN pip3 install --no-cache-dir pytorch-argus
RUN pip3 install --no-cache-dir opencv-python
RUN pip3 install --no-cache-dir tabulate

COPY . /workdir

WORKDIR /workdir
