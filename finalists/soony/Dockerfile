# This is the base Dockerfile for the challenge submission.
# 
# You can edit this Dockerfile as well as the environment.yml file to get the desired runtime environment.
# Don't change the workspace directory defined at the bottom of this file!

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

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

# Define torchvision model cache location
RUN mkdir /torch_model_zoo
RUN chmod -R a+w /torch_model_zoo
ENV TORCH_HOME /torch_model_zoo

RUN pip install numpy pillow spicy sklearn 

# Define the workspace directory
# This is the mount point in which the project directory will mapped at.
# Do not change it!
WORKDIR /workspace
RUN chmod -R a+w .
