# This is the base Dockerfile for the challenge submission.
# 
# You can edit this Dockerfile as well as the environment.yml file to get the desired runtime environment.
# Don't change the workspace directory defined at the bottom of this file!

FROM nvidia/cuda:10.2-runtime-ubuntu18.04

# Install prerequisites
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y \
    wget \
    zip unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# List of apt dependencies:
# wget: miniconda download
# zip, unzip: submission packaging

# Install miniconda3 (from miniconda3 official Dockerfile: https://github.com/ContinuumIO/docker-images/tree/master/miniconda3)
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Create (update) reference conda environment
ADD environment.yml .
RUN conda env update --name base --file environment.yml
RUN rm environment.yml

# Define torchvision model cache location
RUN mkdir /torch_model_zoo
RUN chmod -R a+w /torch_model_zoo
ENV TORCH_HOME /torch_model_zoo

# Define the workspace directory
# This is the mount point in which the project directory will mapped at.
# Do not change it!
WORKDIR /workspace
RUN chmod -R a+w .