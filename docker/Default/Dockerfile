FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04
RUN mkdir /home/src
RUN apt-get update
RUN apt-get install python3 -y && \
    apt-get install python3-pip -y && \
    ln /usr/bin/python3 /usr/bin/python
RUN pip3 install tensorflow-gpu 
RUN DEBIAN_FRONTEND=noninteractive apt-get install python3-opencv -y && \
    apt-get install python3-pandas -y
RUN pip3 install imgaug 