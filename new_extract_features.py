# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import argparse
import os
import sys
import torch
import h5py
import tqdm
import cv2
import numpy as np

sys.path.append('detectron2')

import matplotlib

matplotlib.use('AGG')

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_setup, launch
from detectron2.structures import Instances
from models.bua.layers.nms import nms

from utils.extract_utils import get_image_blob, save_bbox, save_roi_features_by_gt_bbox, save_roi_features
from models import add_config
from models.bua.box_regression import BUABoxes


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(args, cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.BUA.EXTRACTOR.MODE = 1
    default_setup(cfg, args)
    cfg.MODEL.DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    cfg.freeze()
    return cfg


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection2 Inference")
    parser.add_argument(
        "--config-file",
        default="configs/bua-caffe/extract-bua-caffe-r101-fix36.yaml",
        metavar="FILE",
        help="path to config file",
    )

    # --image-dir or --image
    parser.add_argument('--image-dir', dest='image_dir',
                        help='directory with images',
                        default="datasets/demos")
    parser.add_argument('--image', dest='image',
                        help='image')  # e.g. datasets/demos/COCO_val2014_000000060623.jpg
    parser.add_argument("--mode", default="caffe", type=str, help="bua_caffe, ...")
    parser.add_argument('--out-dir', dest='output_dir',
                        help='output directory for features',
                        default="features")
    parser.add_argument('--out-name', dest='output_name',
                        help='output file name for features',
                        default="demos")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    cfg = setup(args)

    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    model.eval()
    # Extract features.
    if args.image:
        imglist = [args.image]
    else:
        imglist = os.listdir(args.image_dir)
        imglist = [os.path.join(args.image_dir, fn) for fn in imglist]
    num_images = len(imglist)
    print('Number of images: {}.'.format(num_images))
    imglist.sort()

    MIN_BOXES = cfg.MODEL.BUA.EXTRACTOR.MIN_BOXES
    MAX_BOXES = cfg.MODEL.BUA.EXTRACTOR.MAX_BOXES
    CONF_THRESH = cfg.MODEL.BUA.EXTRACTOR.CONF_THRESH

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with h5py.File(os.path.join(args.output_dir, '%s_fc.h5' % args.output_name), 'a') as file_fc, \
            h5py.File(os.path.join(args.output_dir, '%s_att.h5' % args.output_name), 'a') as file_att, \
            h5py.File(os.path.join(args.output_dir, '%s_box.h5' % args.output_name), 'a') as file_box:
        informations = {}
        try:
            for im_file in tqdm.tqdm(imglist):
                img_nm = os.path.basename(im_file)
                im = cv2.imread(im_file)
                if im is None:
                    print(im_file, "is illegal!")
                    continue
                dataset_dict = get_image_blob(im, cfg.MODEL.PIXEL_MEAN)
                # extract roi features
                attr_scores = None
                with torch.set_grad_enabled(False):
                    if cfg.MODEL.BUA.ATTRIBUTE_ON:
                        boxes, scores, features_pooled, attr_scores = model([dataset_dict])
                    else:
                        boxes, scores, features_pooled = model([dataset_dict])

                dets = boxes[0].tensor.cpu() / dataset_dict['im_scale']
                scores = scores[0].cpu()
                feats = features_pooled[0].cpu()
                max_conf = torch.zeros((scores.shape[0])).to(scores.device)
                for cls_ind in range(1, scores.shape[1]):
                    cls_scores = scores[:, cls_ind]
                    keep = nms(dets, cls_scores, 0.3)
                    max_conf[keep] = torch.where(cls_scores[keep] > max_conf[keep],
                                                 cls_scores[keep],
                                                 max_conf[keep])

                keep_boxes = torch.nonzero(max_conf >= CONF_THRESH).flatten()
                if len(keep_boxes) < MIN_BOXES:
                    keep_boxes = torch.argsort(max_conf, descending=True)[:MIN_BOXES]
                elif len(keep_boxes) > MAX_BOXES:
                    keep_boxes = torch.argsort(max_conf, descending=True)[:MAX_BOXES]
                image_feat = feats[keep_boxes].numpy()
                image_bboxes = dets[keep_boxes].numpy()
                image_objects_conf = np.max(scores[keep_boxes].numpy()[:, 1:], axis=1)
                image_objects = np.argmax(scores[keep_boxes].numpy()[:, 1:], axis=1)
                info = {
                    'image_name': img_nm,
                    'image_h': np.size(im, 0),
                    'image_w': np.size(im, 1),
                    'num_boxes': len(keep_boxes),
                    'objects_id': image_objects,
                    'objects_conf': image_objects_conf
                }
                if attr_scores is not None:
                    attr_scores = attr_scores[0].cpu()
                    image_attrs_conf = np.max(attr_scores[keep_boxes].numpy()[:, 1:], axis=1)
                    image_attrs = np.argmax(attr_scores[keep_boxes].numpy()[:, 1:], axis=1)
                    info['attrs_id'] = image_attrs
                    info['attrs_conf'] = image_attrs_conf
                file_fc.create_dataset(img_nm, data=image_feat.mean(0))
                file_att.create_dataset(img_nm, data=image_feat)
                file_box.create_dataset(img_nm, data=image_bboxes)
                informations[img_nm] = info
        finally:
            file_fc.close()
            file_att.close()
            file_box.close()
            np.save(os.path.join(args.output_dir, '%s_info.npy' % args.output_name), informations)
            print('--------------------------------------------------------------------')


if __name__ == "__main__":
    main()
