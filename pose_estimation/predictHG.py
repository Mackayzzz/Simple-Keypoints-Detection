from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from collections import OrderedDict
import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import HeatmapLoss, JointsMSELoss
from core.function import train, validateHG
from core.function import validate
from models.hourglass import PoseNet
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger

import dataset
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/coco/hourglass/4stacks_fish.yaml',
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers

def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    model = PoseNet(
        nstack=config.MODEL.N_STACK, inp_dim=256, oup_dim=config.MODEL.NUM_JOINTS
    ).cuda()
    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        state_dict_old = torch.load(config.TEST.MODEL_FILE)
        state_dict = OrderedDict()
        # delete 'module.' because it is saved from DataParallel module
        for key in state_dict_old.keys():
            if key.startswith('module.'):
                # state_dict[key[7:]] = state_dict[key]
                # state_dict.pop(key)
                state_dict[key[7:]] = state_dict_old[key]
            else:
                state_dict[key] = state_dict_old[key]
        model.load_state_dict(state_dict)
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = HeatmapLoss().cuda()

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.'+config.DATASET.DATASET)(
        config,
        config.DATASET.ROOT,
        config.DATASET.TEST_SET,
        False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validateHG(config, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)
    #TODO: 测试用空白测试集转换

if __name__ == '__main__':
    main()
