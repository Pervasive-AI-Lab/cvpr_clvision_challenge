FROM nvidia/cuda:10.2-runtime

RUN mkdir /workspace

WORKDIR /workspace 

COPY . .

RUN apt update && \
    apt install -y --no-install-recommends gcc python3.8 python3.8-dev python3-distutils libglib2.0-dev libsm6 libxrender1 libxext-dev && \
    python3.8 /workspace/get-pip.py && \
    pip3.8 install --default-timeout=100 --no-cache-dir torch==1.4.0 torchvision==0.5.0 scikit-learn imgaug psutil && \
    apt clean && \
    mkdir -p /ram /data /submissions /root/.cache/torch && \
    rm /workspace/get-pip.py

CMD [ "/bin/bash","start_all.sh" ]

# Define torchvision model cache location
RUN mkdir /torch_model_zoo
RUN chmod -R a+w /torch_model_zoo
ENV TORCH_HOME /torch_model_zoo
