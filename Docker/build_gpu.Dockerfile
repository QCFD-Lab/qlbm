FROM nvcr.io/nvidia/cuquantum-appliance:23.10

RUN cat /home/cuquantum/.README | awk '{split($0,a,":"); print a[2]}' |\
    sudo -S apt-get clean \
    && sudo apt-get update \
    && sudo apt-get install -y \
    libboost-all-dev \
    cmake

WORKDIR /quantum-lbm

COPY --chmod=0777 Makefile Makefile
COPY --chmod=0777 pyproject.toml pyproject.toml
COPY --chmod=0777 README.md README.md

RUN make install-gpu

COPY --chmod=0777 qlbm qlbm
COPY --chmod=0777 test test

ENTRYPOINT /bin/bash