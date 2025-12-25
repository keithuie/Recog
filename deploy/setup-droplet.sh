#!/bin/bash
set -e
apt-get update && apt-get install -y docker.io git
cd /opt && git clone https://github.com/keithuie/Recog.git machineiq
cd machineiq && docker build -t miq . && docker run --rm miq
