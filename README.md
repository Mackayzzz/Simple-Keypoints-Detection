# Simple Keypoints Detection


[[1804.06208\] Simple Baselines for Human Pose Estimation and Tracking (arxiv.org)](https://arxiv.org/abs/1804.06208)

[[1603.06937\] Stacked Hourglass Networks for Human Pose Estimation (arxiv.org)](https://arxiv.org/abs/1603.06937)

## Quick start

### Installation
1. 安装python>=3.6; pytorch >= v1.0.0

2. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
   
3. 安装 [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```bash
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

   ```bash
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

   ```bash
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```bash
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
```bash
   ${POSE_ROOT}
   └── data
       └── fish
           ├── annotations
           │   └── fish_7.11_coco.json
           └── images
               └── fish22.6.17
```
### Yaml Setting

experiments/coco/

```yaml
DATASET:
  DATASET: 'coco'
  ROOT: 'data/fish/'                #数据集路径
  FOLDER: 'fish22.6.17'             #图片文件夹
  TEST_SET: 'fish_blank_2400'       #测试label文件名（coco格式）
  TRAIN_SET: 'fish_7.11_coco'       #训练label文件名（coco格式）
```

```yaml
MODEL:
  NAME: 'pose_resnet'               #可选['pose_renet', 'hourglass']
  NUM_JOINTS: 9                     #点的数量
  N_STACK: 8                        #Hourglass堆叠数量
```

```yaml
TEST:
  MODEL_FILE: 'models/pytorch/trained/final_state.pth.tar'  #保存的权重文件路径
```
其余的训练参数均可自行调节

### COCO json

将自己的标签转化成coco格式，预测使用的json可以没有关键点坐标数据，只需要有bbox(检测框)

```json
{
    "images": [
        {
            "file_name": "IMG_20220318_093630.jpg",
            "id": 1,                                    #图片id，每张图片需要不同
            "height": 5616,                             #图片大小
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
                1053.99,                    #检测框左上坐标x
                1056.258,                   #坐标y
                1109.7430000000002,         #宽width
                4851.512000000001           #高height
            ],
            "image_id": 1,                  #图片id
            "category_id": 1,
            "id": 1,
            "area": 5383931.481416002,      #宽×高
            "keypoints": [                  #关键点坐标，若点在图片外第三维为0，
                1525.89,                    若为预测标签，第三维全为2，点坐标为0
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
            "num_keypoints": 9              #点的数量
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
            "keypoints": [                  #修改点名称，其余直接复制
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

## Train

SimpleBaseline
```bash
python pose_estimation/train.py \
    --cfg experiments/coco/resnet152/384x384_fish.yaml
```

StackedHourglass
```bash
python pose_estimation/trainHG.py \
    --cfg experiments/coco/hourglass/4stacks_fish.yaml
```

## Predict
SimpleBaseline
```bash
python pose_estimation/predict.py \
    --cfg experiments/coco/resnet152/384x384_fish.yaml
```

StackedHourglass
```bash
python pose_estimation/predictHG.py \
    --cfg experiments/coco/hourglass/4stacks_fish.yaml
```

自行修改yaml路径

训练和预测的结果保存在output文件夹中

