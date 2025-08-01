FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

RUN conda update conda -y

RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         python3-opencv \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git

RUN mkdir weights && \
    cd weights && \
    wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# CUDA already available from base image; skip reinstallation
# Optional: install extra packages GroundingDINO may need
RUN cd GroundingDINO/ && python -m pip install .

COPY . . 

CMD [ "python3", "groudingdino_pipeline_api.py"]
