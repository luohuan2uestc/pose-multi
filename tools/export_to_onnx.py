# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (leoxiaobin@gmail.com)
# Modified by Bowen Cheng (bcheng9@illinois.edu)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import time
import shutil

from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms
import torch.multiprocessing
import torch.onnx
from tqdm import tqdm
import cv2
import numpy as np

import _init_paths
import models

from config import cfg
from config import check_config
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.group import HeatmapParser
from dataset import make_test_dataloader
from fp16_utils.fp16util import network_to_half
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.vis import save_debug_images
from utils.vis import save_valid_image
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size

torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_output_dirs(prefix='/output/'):
    pose_dir = os.path.join(prefix, "pose")
    if os.path.exists(pose_dir) and os.path.isdir(pose_dir):
        shutil.rmtree(pose_dir)
    os.makedirs(pose_dir, exist_ok=True)
    return pose_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Test keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    
    # parser.add_argument('--videoFile', type=str, required=True)
    
    parser.add_argument('--output_onnx', type=str, default='/pose_hrnet.onnx')
    
    # parser.add_argument('--inferenceFps', type=int, default=10)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    return args


# markdown format output
def _print_name_value(logger, name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values+1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
         ' |'
    )


def main():
    args = parse_args()
    update_config(cfg, args)
    check_config(cfg)
    # pose_dir = prepare_output_dirs(args.outputDir)
    
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid'
    )

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    dump_input = torch.rand(
        (1, 3, cfg.DATASET.INPUT_SIZE, cfg.DATASET.INPUT_SIZE)
    )
    logger.info(get_model_summary(model, dump_input, verbose=cfg.VERBOSE))

    if cfg.FP16.ENABLED:
        model = network_to_half(model)

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=True)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth.tar'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        pretrian_model_state = torch.load(model_state_file)
        
        for name, param in model.state_dict().items():
            
            model.state_dict()[name].copy_(pretrian_model_state['1.'+name])
        # model.load_state_dict(torch.load(model_state_file))

    # model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()
    
    # Input to the model
    # batch_size = 1
    x = torch.randn(1, 3, 256, 256, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    args.output_onnx,   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=11,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output1'] # the model's output names
                    )

if __name__ == '__main__':
    main()
