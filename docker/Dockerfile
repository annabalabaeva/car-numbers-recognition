FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget \
    python3 python3-pip python3-dev python3-setuptools \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

# Build nvidia codec headers
RUN git clone -b sdk/8.2 --single-branch https://git.videolan.org/git/ffmpeg/nv-codec-headers.git &&\
    cd nv-codec-headers && make install &&\
    cd .. && rm -rf nv-codec-headers


# Build ffmpeg with nvenc support
RUN git clone --depth 1 -b release/4.2 --single-branch https://github.com/FFmpeg/FFmpeg.git &&\
    cd FFmpeg &&\
    mkdir ffmpeg_build && cd ffmpeg_build &&\
    ../configure \
    --enable-cuda \
    --enable-cuvid \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-nonfree \
    --enable-gpl \
    --enable-openssl \
    --enable-libx264 \
    --extra-libs=-lpthread \
    --nvccflags="-gencode arch=compute_75,code=sm_75" &&\
    make -j$(nproc) && make install && ldconfig &&\
    cd ../.. && rm -rf FFmpeg

RUN pip3 install --no-cache-dir numpy==1.16.4

RUN pip3 install --no-cache-dir \
    torch==1.2.0 \
    torchvision==0.4.0


ENV PYTHONPATH $PYTHONPATH:/workdir/src
ENV TORCH_HOME=/workdir/data/.torch

ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility

COPY . /workdir

WORKDIR /workdir