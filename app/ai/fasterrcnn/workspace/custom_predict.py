#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import numpy as np
import os
import logging
import shutil
import tensorflow as tf
import cv2
import six
import tqdm
import xml.etree.ElementTree as et

assert six.PY3, "This example requires Python 3!"

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import get_model_loader, get_tf_version_tuple
from tensorpack.utils import fs, logger

from dataset import DatasetRegistry, register_datasets
from config import finalize_configs, config as cfg
from data import get_eval_dataflow, get_train_dataflow
from eval import DetectionResult, multithread_predict_dataflow, predict_image
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from custom_viz import draw_annotation, draw_final_outputs, draw_predictions, draw_proposal_recall

logger = logging.getLogger('custom_predict')
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('[%(asctime)s|%(levelname)s @%(filename)s:%(lineno)d] > %(message)s', 
                    datefmt='%Y/%m/%d %H:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)


def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        logger.info("Delete existing {} dir".format(output_dir))
        shutil.rmtree(output_dir)
    fs.mkdir_p(output_dir)
    logger.info("Images are saved at {} dir".format(output_dir))
    
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels = dp['image'], dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, \
                final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def do_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()

    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        DatasetRegistry.get(dataset).eval_inference_results(all_results, output)


def do_predict(pred_func, input_file, output_dir):
    """"""
    basename = os.path.basename(input_file)
    filename = os.path.splitext(basename)[0]
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    final = draw_final_outputs(img, results)
    
    boxes, labels = _get_annotation(filename, img)
    gt_img = tpviz.draw_boxes(img, boxes, labels)
    viz = np.concatenate((img, gt_img, final), axis=1)
    
#     cv2.imwrite("{}/{}_input.png".format(output_dir, filename), img)
#     cv2.imwrite("{}/{}_pred.png".format(output_dir, filename), final)
    cv2.imwrite("{}/{}_merge.png".format(output_dir, filename), viz)
#     tpviz.interactive_imshow(viz)


def _get_annotation(filename, 
                    img,  
                    annotation_dir='../../../../kidi_data/damage_bbox/valid/annotations'):
    """"""
    xml_path = os.path.join(annotation_dir, '{}.xml'.format(filename))
    logger.debug(xml_path)
    tree = et.parse(xml_path)
    annotation = tree.getroot()
    obj_list = annotation.findall('object')
    h, w, _ = img.shape
    
    boxes = []
    labels = []
    for obj in obj_list:
        labels.append(obj.find("name").text)
        bndbox = obj.find("bndbox")
        xmin = max(0, int(bndbox.find("xmin").text))
        ymin = max(0, int(bndbox.find("ymin").text))
        xmax = min(w, int(bndbox.find("xmax").text))
        ymax = min(h, int(bndbox.find("ymax").text))
        boxes.append([xmin, ymin, xmax, ymax])
    
    return np.array(boxes), np.array(labels)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation.', required=True)
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image dir or file", nargs='+')
    parser.add_argument('--output_dir', help="directory to save output images")
    parser.add_argument('--num_output_images', 
                        help="number of images to predict", 
                        default=10000, type=int)
    parser.add_argument('--benchmark', action='store_true', help="Benchmark the speed of the model + postprocessing")
    parser.add_argument('--config', help="config file path", default=None)
    parser.add_argument('--date', default='', type=str)

    args = parser.parse_args()
    
    if args.config is not None:
        cfg.update_from_json(args.config)
        cfg.DATA.CLASS_NAMES = ['BG'] + cfg.DATA.CLASS_NAMES
        
    logger.debug("cfg.DATA.CLASS_NAMES = {}".format(cfg.DATA.CLASS_NAMES))
        
    register_datasets(cfg.DATA.BASEDIR, args.date)
        
    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util
        assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
            "Inference requires either GPU support or MKL support!"
    assert args.load
    finalize_configs(is_training=False)

    if args.predict or args.visualize:
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    if args.visualize:
        # visualize fpn
        do_visualize(MODEL, args.load)
    else:
        # predict each image
        predcfg = PredictConfig(
            model=MODEL,
            session_init=get_model_loader(args.load),
            input_names=MODEL.get_inference_tensor_names()[0],
            output_names=MODEL.get_inference_tensor_names()[1])
        if args.predict:
            predictor = OfflinePredictor(predcfg)
            # clear output_dir and make new
            if os.path.isdir(args.output_dir):
                logger.info("Delete existing {} dir".format(args.output_dir))
                shutil.rmtree(args.output_dir)
            fs.mkdir_p(args.output_dir)
            if os.path.isdir(args.predict[0]):
                image_dir = args.predict[0]
                image_list = os.listdir(image_dir)
                np.random.shuffle(image_list)
                logger.info("Inference output for {} written to {}".format(image_dir, args.output_dir))
                infer_len = min(args.num_output_images, len(image_list))
                with tqdm.tqdm(total=infer_len) as pbar:
                    for image_file in image_list[:infer_len]:
                        do_predict(predictor, 
                                   os.path.join(image_dir, image_file), 
                                   args.output_dir)
                        pbar.update()
            else:
                image_list = args.predict
                logger.info("Inference output will be written to {}".format(args.output_dir))
                infer_len = min(args.num_output_images, len(image_list))
                
                for image_file in image_list[:infer_len]:
                    do_predict(predictor, image_file, args.output_dir)
            
        elif args.evaluate:
            assert args.evaluate.endswith('.json'), args.evaluate
            do_evaluate(predcfg, args.evaluate)
        elif args.benchmark:
            df = get_eval_dataflow(cfg.DATA.VAL[0])
            df.reset_state()
            predictor = OfflinePredictor(predcfg)
            for img in tqdm.tqdm(df, total=len(df)):
                # This include post-processing time, which is done on CPU and not optimized
                # To exclude it, modify `predict_image`.
                predict_image(img[0], predictor)
