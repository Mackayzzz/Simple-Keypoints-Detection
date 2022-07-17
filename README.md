# Simple Keypoints Detection

## Quick start
### Installation
1. 安装python>=3.6; pytorch >= v1.0.0 following.

2. 安装依赖:
   ```
   pip install -r requirements.txt
   ```
   
3. 安装 [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
   
4. 下载Pytorch resnet预训练模型. 

    [resnet50](http://download.pytorch.org/models/resnet50-19c8e357.pth)

    [resnet101](http://download.pytorch.org/models/resnet101-5d3b4d8f.pth)

    [resnet152](http://download.pytorch.org/models/resnet152-b121ed2d.pth)

   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            `-- imagenet
                |-- resnet50-19c8e357.pth
                |-- resnet101-5d3b4d8f.pth
                `-- resnet152-b121ed2d.pth
   
   ```

5. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── modelfile
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```

### Data preparation


### Yaml Setting





### Predicting 

```
python pose_estimation/valid.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml \
    --flip-test \
    --model-file models/pytorch/pose_mpii/pose_resnet_50_256x256.pth.tar
```

### Training on MPII

```
python pose_estimation/train.py \
    --cfg experiments/mpii/resnet50/256x256_d256x3_adam_lr1e-3.yaml
```

### Valid on COCO val2017 using pretrained models

```
python pose_estimation/predict.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml
```

### Training

```
python pose_estimation/train.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml
```
