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

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

def get_pose_estimation_prediction(pose_model, image, centers, scales, transform):
    rotation = 0

    # pose estimation transformation
    model_inputs = []
    for center, scale in zip(centers, scales):
        trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
        # Crop smaller image of people
        model_input = cv2.warpAffine(
            image,
            trans,
            (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
            flags=cv2.INTER_LINEAR)

        # hwc -> 1chw
        model_input = transform(model_input)#.unsqueeze(0)
        model_inputs.append(model_input)

    # n * 1chw -> nchw
    model_inputs = torch.stack(model_inputs)

    # compute output heatmap
    output = pose_model(model_inputs.to(CTX))
    coords, _ = get_final_preds(
        cfg,
        output.cpu().detach().numpy(),
        np.asarray(centers),
        np.asarray(scales))

    return coords

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
    
    parser.add_argument('--videoFile', type=str, required=True)
    
    parser.add_argument('--outputDir', type=str, default='/output/')
    
    parser.add_argument('--inferenceFps', type=int, default=10)

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
    pose_dir = prepare_output_dirs(args.outputDir)
    
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
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    model.eval()

    # data_loader, test_dataset = make_test_dataloader(cfg)
    if cfg.MODEL.NAME == 'pose_hourglass':
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )
    else:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

    parser = HeatmapParser(cfg)
    # Loading an video
    vidcap = cv2.VideoCapture(args.videoFile)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    if fps < args.inferenceFps:
        print('desired inference fps is '+str(args.inferenceFps)+' but video fps is '+str(fps))
        exit()
    skip_frame_cnt = round(fps / args.inferenceFps)
    frame_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    outcap = cv2.VideoWriter('{}/{}_pose.avi'.format(args.outputDir, os.path.splitext(os.path.basename(args.videoFile))[0]),
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), int(skip_frame_cnt), (frame_width, frame_height))

    count = 0
    while vidcap.isOpened():
        total_now = time.time()
        ret, image_bgr = vidcap.read()
        count += 1

        if not ret:
            continue

        if count % skip_frame_cnt != 0:
            continue
        
        image_debug = image_bgr.copy()
        now = time.time()
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # image = image_rgb.cpu().numpy()
        # size at scale 1.0
        base_size, center, scale = get_multi_scale_size(
            image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
        )
        with torch.no_grad():
            final_heatmaps = None
            tags_list = []
            for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
                input_size = cfg.DATASET.INPUT_SIZE
                image_resized, center, scale = resize_align_multi_scale(
                    image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
                )
                image_resized = transforms(image_resized)
                image_resized = image_resized.unsqueeze(0).cuda()

                outputs, heatmaps, tags = get_multi_stage_outputs(
                    cfg, model, image_resized, cfg.TEST.FLIP_TEST,
                    cfg.TEST.PROJECT2IMAGE, base_size
                )

                final_heatmaps, tags_list = aggregate_results(
                    cfg, s, final_heatmaps, tags_list, heatmaps, tags
                )

            final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
            tags = torch.cat(tags_list, dim=4)
            grouped, scores = parser.parse(
                final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
            )

            final_results = get_final_preds(
                grouped, center, scale,
                [final_heatmaps.size(3), final_heatmaps.size(2)]
            )
        for person_joints in final_results:
            for joint in person_joints:
                x, y = int(joint[0]), int(joint[1])
                cv2.circle(image_debug, (x, y), 4, (255, 0, 0), 2)
        then = time.time()
        print("Find person pose in: {} sec".format(then - now))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_file = os.path.join(pose_dir, 'pose_{:08d}.jpg'.format(count))
        cv2.imwrite(img_file, image_debug)
        outcap.write(image_debug)

    
    vidcap.release()
    outcap.release()

    # if cfg.MODEL.NAME == 'pose_hourglass':
    #     transforms = torchvision.transforms.Compose(
    #         [
    #             torchvision.transforms.ToTensor(),
    #         ]
    #     )
    # else:
    #     transforms = torchvision.transforms.Compose(
    #         [
    #             torchvision.transforms.ToTensor(),
    #             torchvision.transforms.Normalize(
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225]
    #             )
    #         ]
    #     )

    # parser = HeatmapParser(cfg)
    # all_preds = []
    # all_scores = []

    # pbar = tqdm(total=len(test_dataset)) if cfg.TEST.LOG_PROGRESS else None
    # for i, (images, annos) in enumerate(data_loader):
    #     assert 1 == images.size(0), 'Test batch size should be 1'

    #     image = images[0].cpu().numpy()
    #     # size at scale 1.0
    #     base_size, center, scale = get_multi_scale_size(
    #         image, cfg.DATASET.INPUT_SIZE, 1.0, min(cfg.TEST.SCALE_FACTOR)
    #     )

    #     with torch.no_grad():
    #         final_heatmaps = None
    #         tags_list = []
    #         for idx, s in enumerate(sorted(cfg.TEST.SCALE_FACTOR, reverse=True)):
    #             input_size = cfg.DATASET.INPUT_SIZE
    #             image_resized, center, scale = resize_align_multi_scale(
    #                 image, input_size, s, min(cfg.TEST.SCALE_FACTOR)
    #             )
    #             image_resized = transforms(image_resized)
    #             image_resized = image_resized.unsqueeze(0).cuda()

    #             outputs, heatmaps, tags = get_multi_stage_outputs(
    #                 cfg, model, image_resized, cfg.TEST.FLIP_TEST,
    #                 cfg.TEST.PROJECT2IMAGE, base_size
    #             )

    #             final_heatmaps, tags_list = aggregate_results(
    #                 cfg, s, final_heatmaps, tags_list, heatmaps, tags
    #             )

    #         final_heatmaps = final_heatmaps / float(len(cfg.TEST.SCALE_FACTOR))
    #         tags = torch.cat(tags_list, dim=4)
    #         grouped, scores = parser.parse(
    #             final_heatmaps, tags, cfg.TEST.ADJUST, cfg.TEST.REFINE
    #         )

    #         final_results = get_final_preds(
    #             grouped, center, scale,
    #             [final_heatmaps.size(3), final_heatmaps.size(2)]
    #         )

    #     if cfg.TEST.LOG_PROGRESS:
    #         pbar.update()

    #     if i % cfg.PRINT_FREQ == 0:
    #         prefix = '{}_{}'.format(os.path.join(final_output_dir, 'result_valid'), i)
    #         # logger.info('=> write {}'.format(prefix))
    #         save_valid_image(image, final_results, '{}.jpg'.format(prefix), dataset=test_dataset.name)
    #         # save_debug_images(cfg, image_resized, None, None, outputs, prefix)

    #     all_preds.append(final_results)
    #     all_scores.append(scores)

    # if cfg.TEST.LOG_PROGRESS:
    #     pbar.close()

    # name_values, _ = test_dataset.evaluate(
    #     cfg, all_preds, all_scores, final_output_dir
    # )

    # if isinstance(name_values, list):
    #     for name_value in name_values:
    #         _print_name_value(logger, name_value, cfg.MODEL.NAME)
    # else:
    #     _print_name_value(logger, name_values, cfg.MODEL.NAME)


if __name__ == '__main__':
    main()
