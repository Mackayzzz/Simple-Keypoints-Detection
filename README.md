# Simple Keypoints Detection

暂仅支持Simple Baselines模型

[论文地址]([[1804.06208\] Simple Baselines for Human Pose Estimation and Tracking (arxiv.org)](https://arxiv.org/abs/1804.06208))

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
            |-- imagenet
            |   |-- resnet50-19c8e357.pth
            |   |-- resnet101-5d3b4d8f.pth
            |   `-- resnet152-b121ed2d.pth
            `-- trained
   
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
   │   └── fish
   │       ├── annotations
   │       │   └── fish_7.11_coco.json
   │       └── images
   │           └── fish22.6.17
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── pose_estimation
   ├── README.md
   └── requirements.txt
   ```

### Data preparation


### Yaml Setting

experiments/coco/

```
DATASET:
  DATASET: 'coco'
  ROOT: 'data/fish/'				#数据集路径
  FOLDER: 'fish22.6.17'				#图片文件夹
  TEST_SET: 'fish_blank_2400'		#测试label文件名（coco格式）
  TRAIN_SET: 'fish_7.11_coco'		#训练label文件名（coco格式）
```

```
MODEL:
  NUM_JOINTS: 9						#点的数量
```

```
TEST:
  MODEL_FILE: 'models/pytorch/trained/final_state.pth.tar' #保存的权重文件路径
```

### COCO json

将自己的标签转化成coco格式，预测使用的json可以没有关键点坐标数据，只需要有bbox(检测框)

```
{
    "images": [
        {
            "file_name": "IMG_20220318_093630.jpg",
            "id": 1,									#图片id，每张图片需要不同
            "height": 5616,								#图片大小
            "width": 2592
        },
        ·
        ·
        ·
    ],
        "annotations": [
        {
            "iscrowd": 0,
            "bbox": [
                1053.99,								#检测框左上坐标x
                1056.258,								#y
                1109.7430000000002,						#长
                4851.512000000001						#宽
            ],
            "image_id": 1,								#图片id
            "category_id": 1,
            "id": 1,
            "area": 5383931.481416002,					#长×宽
            "keypoints": [								#关键点坐标，若点在图片外第三维为0，若为预测标签，第三维全为2，点坐标随意
                1525.89,
                1173.62,
                2,
                1662.9,
                2066.9,
                2,
                1453.1,
                2058.0,
                2,
                1967.03,
                ·
                ·
                ·
            ],
            "num_keypoints": 9							#点的数量
        },
        ·
        ·
        ·
        ·
    ],
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [								#修改点名称，其余直接复制
                "Eh",
                "Et"
            ],
            "skeleton": [
                [
                    16,
                    14
                ],
                [
                    14,
                    12
                ],
                [
                    17,
                    15
                ],
                [
                    15,
                    13
                ],
                [
                    12,
                    13
                ],
                [
                    6,
                    12
                ],
                [
                    7,
                    13
                ],
                [
                    6,
                    7
                ],
                [
                    6,
                    8
                ],
                [
                    7,
                    9
                ],
                [
                    8,
                    10
                ],
                [
                    9,
                    11
                ],
                [
                    2,
                    3
                ],
                [
                    1,
                    2
                ],
                [
                    1,
                    3
                ],
                [
                    2,
                    4
                ],
                [
                    3,
                    5
                ],
                [
                    4,
                    6
                ],
                [
                    5,
                    7
                ]
            ]
        }
    ]
}
```

### Train

```
python pose_estimation/train.py \
    --cfg experiments/coco/resnet152/384x384_fish.yaml
```

### Predict

```
python pose_estimation/predict.py \
    --cfg experiments/coco/resnet152/384x384_fish.yaml
```

自行修改yaml路径

训练和预测的结果保存在output文件夹中

