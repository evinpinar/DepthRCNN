[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
# Depth Prediction from MaskRCNN

This repository is forked from PlaneRCNN and has being modified for own purposes. 


------------


## Introduction

## Getting Started 
Clone repository: 
```
git clone https://github.com/NVlabs/depthrcnn.git
```

Please use Python 3. Create an [Anaconda](https://www.anaconda.com/distribution/) environment and install the dependencies:
```
conda create --name planercnn
conda activate planercnn
conda install -y pytorch=0.4.1
conda install pip
pip install -r requirements.txt
```
Equivalently, you can use Python virtual environment to manage the dependencies:
```
pip install virtualenv
python -m virtualenv planercnn
source planercnn/bin/activate
pip install -r requirements.txt
```
Now, we compile nms and roialign as explained in the installation section of [pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn). To be specific, you can build these two functions using the following commands with the right `--arch` option:

 | GPU                     | arch  |
 |-------------------------|-------|
 | TitanX                  | sm_52 |
 | GTX 960M                | sm_50 |
 | GTX 1070                | sm_61 |
 | GTX 1080 (Ti), Titan XP | sm_61 |

More details of the compute capability are shown in [NVIDIA](https://developer.nvidia.com/cuda-gpus)

```bash
cd nms/src/cuda/
nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cd ../


cd roialign/roi_align/src/cuda/
nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
cd ../../
python build.py
cd ../../

```
Please note that, the Mask R-CNN backbone does not support cuda10.0 and gcc versions higher than 6.




## Acknowledgement
Our implementation uses the nms/roialign from the Mask R-CNN implementation from [pytorch-mask-rcnn](https://github.com/multimodallearning/pytorch-mask-rcnn), which is licensed under [MIT License](https://github.com/multimodallearning/pytorch-mask-rcnn/blob/master/LICENSE)

### License ###
Copyright (c) 2018 NVIDIA Corp.  All Rights Reserved.
This work is licensed under the [Creative Commons Attribution NonCommercial ShareAlike 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

