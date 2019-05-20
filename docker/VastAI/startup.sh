#!/bin/bash
docker run --mount source="$PWD/../src",target="/home/src",type=bind --rm --name c_ros3_dev -it --runtime=nvidia rov3_ssd /bin/bash