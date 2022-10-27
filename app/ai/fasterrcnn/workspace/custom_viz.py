# -*- coding: utf-8 -*-
# File: viz.py

import logging
import numpy as np
from six.moves import zip

from tensorpack.utils import viz
# from tensorpack.utils.palette import PALETTE_RGB

from config import config as cfg
from utils.np_box_ops import area as np_area
from utils.np_box_ops import iou as np_iou

logger = logging.getLogger('custom_viz')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(asctime)s|%(levelname)s @%(name)s > %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

PARTS_MASK_RGB_DICT = {
    'back_door': [50, 50, 150],
    'front_bumper': [250, 50, 250],
    'front_door_left': [250, 250, 50],
    'front_door_right': [150, 150, 250],
    'front_fender_left': [250, 50, 150],
    'front_fender_right': [250, 150, 250],
    'front_fog_left': [150, 50, 150],
    'front_fog_right': [150, 150, 150],
    'front_lamp_left': [50, 50, 250],
    'front_lamp_right': [250, 150, 150],
    'grille_up': [250, 250, 150],
    'hood': [250, 250, 250],
    'rear_bumper': [250, 50, 50],
    'rear_door_left': [150, 150, 50],
    'rear_door_right': [50, 250, 250],
    'rear_fender_left': [150, 50, 50],
    'rear_fender_right': [150, 250, 150],
    'rear_lamp_left': [50, 50, 50],
    'rear_lamp_right': [50, 150, 50],
    'rear_stop_center': [50, 150, 150],
    'rear_stop_left': [50, 250, 50],
    'rear_stop_right': [250, 150, 50],
    'side_mirror_left': [150, 50, 250],
    'side_mirror_right': [150, 250, 50],
    'side_step_left': [50, 150, 250],
    'side_step_right': [150, 250, 250],
    'trunk': [50, 250, 150]
}


def draw_annotation(img, boxes, klass, is_crowd=None):
    """Will not modify img"""
    labels = []
    assert len(boxes) == len(klass)
    if is_crowd is not None:
        assert len(boxes) == len(is_crowd)
        for cls, crd in zip(klass, is_crowd):
            clsname = cfg.DATA.CLASS_NAMES[cls]
            if crd == 1:
                clsname += ';Crowd'
            labels.append(clsname)
    else:
        for cls in klass:
            labels.append(cfg.DATA.CLASS_NAMES[cls])
    img = viz.draw_boxes(img, boxes, labels)
    return img


def draw_proposal_recall(img, proposals, proposal_scores, gt_boxes):
    """
    Draw top3 proposals for each gt.
    Args:
        proposals: NPx4
        proposal_scores: NP
        gt_boxes: NG
    """
    box_ious = np_iou(gt_boxes, proposals)    # ng x np
    box_ious_argsort = np.argsort(-box_ious, axis=1)
    good_proposals_ind = box_ious_argsort[:, :3]   # for each gt, find 3 best proposals
    good_proposals_ind = np.unique(good_proposals_ind.ravel())

    proposals = proposals[good_proposals_ind, :]
    tags = list(map(str, proposal_scores[good_proposals_ind]))
    img = viz.draw_boxes(img, proposals, tags)
    return img, good_proposals_ind


def draw_predictions(img, boxes, scores):
    """
    Args:
        boxes: kx4
        scores: kxC
    """
    if len(boxes) == 0:
        return img
    labels = scores.argmax(axis=1)
    scores = scores.max(axis=1)
    tags = ["{},{:.2f}".format(cfg.DATA.CLASS_NAMES[lb], score) for lb, score in zip(labels, scores)]
    return viz.draw_boxes(img, boxes, tags)


def draw_final_outputs(img, results):
    """
    Args:
        results: [DetectionResult]
    """
    if len(results) == 0:
        return img

    # Display in largest to smallest order to reduce occlusion
    boxes = np.asarray([r.box for r in results])
    areas = np_area(boxes)
    sorted_inds = np.argsort(-areas)

    ret = img
    tags = []

    for result_id in sorted_inds:
        r = results[result_id]
        logger.debug("results[result_id] = {}".format(r))
        if r.mask is not None:
            c_name = cfg.DATA.CLASS_NAMES[r.class_id]
            color = np.array(PARTS_MASK_RGB_DICT[c_name]).astype(np.float)
            ret = draw_mask(ret, r.mask, color=color)
    logger.debug('len cfg.DATA.CLASS_NAMES = {}'.format(len(cfg.DATA.CLASS_NAMES)))
    for r in results:
        logger.debug('r.class_id = {}'.format(r.class_id))
        tags.append(
            "{},{:.2f}".format(cfg.DATA.CLASS_NAMES[r.class_id], r.score))
    ret = viz.draw_boxes(ret, boxes, tags)
    return ret


def draw_mask(im, mask, alpha=0.5, color=None):
    """
    Overlay a mask on top of the image.

    Args:
        im: a 3-channel uint8 image in BGR
        mask: a binary 1-channel image of the same size
        color: if None, will choose automatically
    """
    if color is None:
        color = PALETTE_RGB[np.random.choice(len(PALETTE_RGB))][::-1]
    im = np.where(np.repeat((mask > 0)[:, :, None], 3, axis=2),
                  im * (1 - alpha) + color * alpha, im)
    im = im.astype('uint8')
    return im
