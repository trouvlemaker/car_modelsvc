import json, time, os
from pprint import pprint
from collections import namedtuple
import matplotlib.pyplot as plt

from app.config import config

from app.ai.utils import *
from app.ai.tools import *
from app.ai.config import *
from app.ai.fasterrcnn.common import CustomResize, clip_boxes
from app.ai.fasterrcnn.eval import _paste_mask

from tensorrtserver.api import ProtocolType, InferContext, InferenceServerException

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logger = create_logger(__name__)


def filter_cls_detection(data, image_out=True):
    # print(hm.check('[{}] Before filter predict'.format(threading.get_ident())), flush=True)

    inference_time = 0
    # logger.debug('filter cls dection : %s' % image_path)
    # Get Info
    image = data["data_res_out"]["image"]
    shape = model_option["filter"]["shape"]
    fast_pass_list = FILTER_DF.loc[FILTER_DF["fast_pass"] == True].index.values

    mobile_class_list = FILTER_DF["mobile_class"].values
    mobile_class_list = mobile_class_list[~(mobile_class_list == None)]

    if data["mobile_info"]["check"]:
        mobile_class = check_valid(data["mobile_info"], "class")
        mobile_part_name = check_valid(data["mobile_info"], "part_name")

        if mobile_part_name == "mobile_etc":
            mobile_class = "mobile_etc"

        if mobile_class in mobile_class_list :
            mobile_class = FILTER_DF.loc[FILTER_DF["mobile_class"] == mobile_class].index.values.item()

    else:
        mobile_class = None

    # Pre-Process
#     input_image = crop_or_pad(image, (shape, shape))
    input_image = cv2.resize(image, (shape, shape))
    input_image = np.expand_dims(input_image, 0).astype(np.float32)

    if np.nanmax(input_image) > 1 :
        input_image = input_image/255.
#         input_image = np.clip(np.multiply(input_image, 255), 0, 255).astype(np.float32)

    # Check orc, dashboard, vincode
    if mobile_class in fast_pass_list:
        probs = 0
        score = 0
        model_cls = FILTER_DF.loc[FILTER_DF.index == mobile_class, "class"].values[0]
        filter_code = filter_class_to_code(model_cls=model_cls)
    else:
        start = time.time()
        model_name = '{}'.format(config['FILTER_MODEL'])
        # model_name = '{}-{}'.format(config['FILTER_MODEL'], config['MODEL_VERSION'])
        url = config['FILTER_SVC']
        print(input_image.shape)
        probs = inception_inference_trtserv(url, model_name, input_image, True)
        stop = time.time()
        cal_time = stop-start
        logger.debug('Prediction completed : {} : {}'.format(cal_time, 'filter-model'))
        inference_time = round(cal_time, 2)

        # Post-Process
        probs = np.round(np.squeeze(probs), 4)
        score = np.max(probs)
        model_cls = np.argmax(probs)
        filter_code = filter_class_to_code(model_cls=model_cls)

    # Make report
    if image_out:
        report = {"check": model_cls == 0,
                  "image": np.squeeze(input_image),
                  "probs": probs,
                  "score": score,
                  "class": model_cls,
                  "code": filter_code
                  }
    else:
        report = {"check": model_cls == 0,
                  "probs": probs,
                  "score": score,
                  "class": model_cls,
                  "code": filter_code
                  }

    # del(input_image)
    return report, inference_time


def parts_detection(data, model_name=None, model_svc=None, masking=False, masking_output_path=None):
    inference_time = 0
    # logger.debug('parts detection serve')
    image_data = data['data_res_out']

    #
    # Pre-Processing
    input_image = [resize_image(image_data['image'],
                                min_dim=MASK_CONFIG.IMAGE_MIN_DIM,
                                max_dim=MASK_CONFIG.IMAGE_MAX_DIM,
                                min_scale=0)[0]]

    # Inference
    start = time.time()
    if model_name is None:
        model_name = '{}'.format(config['PARTS_MODEL'])
        # model_name = '{}-{}'.format(config['PARTS_MODEL'], config['MODEL_VERSION'])
        model_svc = model_svc
    part_res_out = parts_inference_trtserv(model_name, input_image, model_svc)[0]
    stop = time.time()
    cal_time = stop - start
    logger.debug('Seldon prediction completed : {} : {}'.format(cal_time, 'parts-model'))
    inference_time = round(cal_time, 2)

    start = time.time()

    # Get info
    thresh = model_option["parts"]["score_thres"]
    aos_part_name = check_valid(check_valid(data, "aos_info"), "part_name")
    aos_part_group = check_valid(check_valid(data, "aos_info"), "part_group")
    mobile_part_name = check_valid(check_valid(data, "mobile_info"), "part_name")
    mobile_tiny = check_valid(check_valid(data, "mobile_info"), "tiny")

    # Post-Processing
    part_res_out['part_name'] = np.array([PARTS_DF.loc[PARTS_DF["model_cls"] == x, "part_name"].values[0] \
                                              for x in part_res_out['class_ids']])

    ## Merge Rear Lamp
    for lr_sep in ["left", "right"]:
        check_list = np.array(["_".join(["rear_lamp", lr_sep]), "_".join(["rear_stop", lr_sep])])
        part_checker = []
        score_checker = []

        for part_name in check_list:
            _part_check = part_name in part_res_out['part_name']
            if _part_check:
                _score_check = part_res_out['scores'][np.where(part_res_out['part_name'] == part_name)[0][0]].item() >= thresh
            else:
                _score_check = False

            part_checker.append(_part_check)
            score_checker.append(_score_check)

        checker = np.array(part_checker) * np.array(score_checker)

        ### Merge rear_stop & rear_lamp into rear_lamp
        if np.sum(checker) == 2:

            lamp_idx = np.where(part_res_out['part_name'] == check_list[0])[0][0]
            stop_idx = np.where(part_res_out['part_name'] == check_list[1])[0][0]

            ### merge output
            part_res_out['scores'][lamp_idx] = np.nanmean(part_res_out['scores'][np.array([lamp_idx, stop_idx])])
            part_res_out['masks'][..., lamp_idx] = np.sum(part_res_out['masks'][..., np.array([lamp_idx, stop_idx])], axis=-1).astype(np.bool)

            ### remove stop lamp
            keep_idx = ~(part_res_out['part_name'] == check_list[1])

            part_res_out['rois'] = part_res_out['rois'][keep_idx]
            part_res_out['class_ids'] = part_res_out['class_ids'][keep_idx]
            part_res_out['scores'] = part_res_out['scores'][keep_idx]
            part_res_out['masks'] = part_res_out['masks'][..., keep_idx]
            part_res_out['part_name'] = part_res_out['part_name'][keep_idx]

        ### replace rear_stop to rear_lamp if detected stop only
        elif (np.sum(checker) == 1) & (checker[1] == True):
            stop_idx = np.where(part_res_out['part_name'] == check_list[1])[0]

            part_res_out['class_ids'][stop_idx] = PARTS_DF.loc[PARTS_DF['part_name'] == check_list[0], "model_cls"].values[0]
            part_res_out['part_name'][stop_idx] = check_list[0]

    # Make output
    if len(part_res_out['class_ids']) > 0:
        thresh_idx = part_res_out['scores'] >= thresh
        dup_mask = np.nansum(part_res_out['masks'][..., thresh_idx], axis=-1) > 1

    output_list = []
    for idx in range(len(part_res_out['class_ids'])):
        mask = part_res_out['masks'][..., idx]
        class_ids = part_res_out['class_ids'][idx]
        score = part_res_out['scores'][idx]
        part_name = part_res_out['part_name'][idx]

        # Condition handling
        if "side_mirror" in part_name:
            pass
        elif PARTS_DF.loc[part_name, "model_name"] is None:
            continue

        # Mask expand
        if class_ids not in []:
            mask_expanded = mask_expand(mask)
        else:
            mask_expanded = mask

        # Mask resize
        resized_mask = mask_reshape(mask=mask_expanded, shape=data["data_res_out"]["shape"])

        # apply mask
        image_masked = input_image[0].copy()
        image_masked[np.where(mask_expanded == False)] = np.array([0, 0, 0], dtype=np.uint8)

        # Check rule minimun-size
        _rule_minsize = False

        # Check rule close
        _rule_close = rule_close(resized_mask)

        # Check rule focus
        _rule_focus, inbox_ratio = rule_focus(resized_mask)

        ## one area multiple parts detected
        _rule_dup = rule_dup(mask=mask, dup_mask=dup_mask)

        ## Not clean mask detected
        _rule_cont = rule_cont(mask=mask)

        ## Final rule checker
        _rule_out = np.any([_rule_minsize, _rule_focus, _rule_close, _rule_dup, _rule_cont, score < thresh])

        # make output
        res_out = dict()

        res_out["part_name"]     = [part_name]
        res_out["ids"]           = [class_ids]
        res_out["score"]         = [np.round(score, 4)]
        res_out["mask"]          = [mask]
        res_out["mask_expanded"] = [mask_expanded]
        res_out["resized_mask"]  = [resized_mask]
        res_out["mask_ratio"]    = [np.nanmean(mask)]
        res_out["image_masked"]  = [image_masked]
        res_out["inbox_ratio"]   = [inbox_ratio]
        res_out["rule_minsize"]  = [_rule_minsize]
        res_out["rule_focus"]    = [_rule_focus]
        res_out["rule_close"]    = [_rule_close]
        res_out["rule_dup"]      = [_rule_dup]
        res_out["rule_cont"]     = [_rule_cont]
        res_out["rule_out"]      = [_rule_out]

        output_list.append(pd.DataFrame(res_out))

    # Make output df
    output_df = pd.concat(output_list).reset_index(drop=True)

    # One part multiple detected
    output_df["select_score"] = output_df["score"] + output_df["mask_ratio"] + \
    output_df["inbox_ratio"] + output_df["rule_out"].astype("int")

    output_df["select_rank"] = output_df.groupby("part_name")["select_score"].rank("dense", ascending=False).astype(int)
    output_df = output_df.loc[output_df["select_rank"] == 1].drop(["select_rank"], axis=1).reset_index(drop=True)

    # Check part_group replacement
    if (aos_part_group is True) & (aos_part_name is not None) & (aos_part_name != "DGPTXX") :
        output_df["group_replace"] = False

        ## check group first
        target_part_grp = PARTS_DF.loc[PARTS_DF["part_name"] == aos_part_name, "part_group"].values
        target_part_grp = None if len(target_part_grp) == 0 else target_part_grp[0]

        if target_part_grp is not None :
            check_part_group_list = PARTS_DF.loc[PARTS_DF["part_group"] == target_part_grp, "part_name"].values

            output_df.loc[(output_df["rule_out"]) & \
                          (output_df["part_name"].apply(lambda x : x in check_part_group_list)), "group_replace"] = True

        ## check upper group
        if not output_df["group_replace"].any() :
            target_part_grp_upper = PARTS_DF.loc[PARTS_DF["part_name"] == aos_part_name, "part_group_upper"].values
            target_part_grp_upper = None if len(target_part_grp_upper) == 0 else target_part_grp_upper[0]

            if target_part_grp_upper is not None :
                check_part_group_upper_list = PARTS_DF.loc[PARTS_DF["part_group_upper"] == target_part_grp_upper, "part_name"].values

                output_df.loc[(output_df["rule_out"]) & \
                              (output_df["part_name"].apply(lambda x : x in check_part_group_upper_list)), "group_replace"] = True

        ## group replacement
        if output_df["group_replace"].sum() > 0 :
            output_df["replace_select_rank"] = output_df.groupby("group_replace")["select_score"].rank("dense", ascending=False).astype(int)
            output_df.loc[~((output_df["group_replace"] == True) & (output_df["replace_select_rank"] == 1)), "group_replace"] = False

            group_replace_idx = output_df.loc[output_df["group_replace"]].index.values[0]
            output_df.loc[output_df.index == group_replace_idx, "part_name"] = aos_part_name
            output_df.loc[output_df.index == group_replace_idx, "ids"] = PARTS_DF.loc[PARTS_DF["part_name"] == aos_part_name, "model_cls"].values[0]
            output_df.loc[output_df.index != group_replace_idx, "group_replace"] = False

            ## check dup
            output_df["select_rank"] = output_df.groupby("part_name")["select_score"].rank("dense", ascending=False).astype(int)
            output_df = output_df.loc[output_df["select_rank"] == 1].drop(["replace_select_rank", "select_rank"], axis=1).reset_index(drop=True)

    # Final Rule check
    output_df["rule_adjust"] = False
    output_df.loc[output_df["part_name"] == aos_part_name, "rule_out"] = False
    output_df.loc[output_df["part_name"] == aos_part_name, "rule_adjust"] = True

    # Make adjustable
    output_df["adjustable"] = output_df["rule_out"]
    if aos_part_name == "AOS_EXCLUDE" :
        output_df["adjustable"] = False

    # Mobile tiny process
    if (mobile_part_name is not None) & (mobile_part_name != "mobile_etc") & (mobile_tiny is True) :
        rule_out_checker = output_df.loc[output_df["part_name"] == mobile_part_name, "rule_out"].values
        select_idx = output_df.loc[output_df["select_score"] == output_df["select_score"].max()].index.values

        # rule free pass
        if (len(rule_out_checker) > 0) & (any(rule_out_checker == True)) :
            output_df.loc[output_df["part_name"] == mobile_part_name, "rule_out"] = False
            output_df.loc[output_df["part_name"] == mobile_part_name, "rule_adjust"] = True

        # if part not exsits
        elif (len(rule_out_checker) == 0) & (len(select_idx) > 0) :
            select_idx = select_idx[0]
            output_df.loc[output_df.index == select_idx, "part_name"] = mobile_part_name
            output_df.loc[output_df.index == select_idx, "ids"] = PARTS_DF.loc[PARTS_DF["part_name"] == mobile_part_name, "model_cls"].values[0]
            output_df.loc[output_df.index == select_idx, "rule_out"] = False
            output_df.loc[output_df.index == select_idx, "rule_adjust"] = True
            output_df.loc[output_df.index == select_idx, "group_replace"] = True

    # Make report
    part_res = output_df.drop(["part_name", "mask_ratio", "select_score"], axis=1)
    part_res.index = output_df["part_name"]
    part_res = part_res.T.to_dict()

    if masking:
        if len(part_res.keys()) > 0:
            mask_for_tool = np.ones(image_data['shape'], dtype=np.uint8)

            for part_name in part_res.keys():
                if part_res[part_name]["score"] >= thresh:
                    mask_for_tool[np.where(part_res[part_name]["resized_mask"] == True)] = np.array(
                        PARTS_DF.loc[part_name, "rgb"], dtype=np.uint8)

            plt.figure(figsize=(12, 12))
            plt.imsave(masking_output_path, mask_for_tool)

            report = {"check": len(part_res) > 0,
                      "mask_output": masking_output_path,
                      "mask_score": np.nanmin(part_res_out["scores"][thresh_idx])
                      }
        else:
            mask_for_tool = None
            report = {"check": False,
                      "mask_output": ""
                      }
    else:

        report = {"check": len(part_res) > 0,
                  "report": part_res
                  }

    return report, inference_time


def damage_detection(data, model_name=None, model_svc=None):
    # print(hm.check('[{}] Before damage predict'.format(threading.get_ident())), flush=True)

    inference_time = 0
    DetectionResult = namedtuple(
        'DetectionResult',
        ['box', 'score', 'class_id', 'mask'])

    image = data["data_res_out"]["image"]
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    orig_shape = data["data_res_out"]["shape"][:2]
    thresh = model_option["damage"]["score_thres"]

    # Preprocess
    TEST_SHORT_EDGE_SIZE = model_option["damage"]["TEST_SHORT_EDGE_SIZE"]
    MAX_SIZE = model_option["damage"]["MAX_SIZE"]
    resizer = CustomResize(TEST_SHORT_EDGE_SIZE, MAX_SIZE)
    resized_img = resizer.augment(img)
    scale = np.sqrt(resized_img.shape[0] * 1.0 / img.shape[0] * resized_img.shape[1] / img.shape[1])

    # Get output
    start = time.time()
    # model_name = 'tensorrt-damage-model'
    if model_name is None:
        model_name = '{}'.format(config['DAMAGE_MODEL'])
        # model_name = '{}-{}'.format(config['DAMAGE_MODEL'], config['MODEL_VERSION'])
        model_svc = model_svc
    boxes, probs, labels, *masks = damage_inference_trtserv(model_name, resized_img, model_svc)
    stop = time.time()
    cal_time = stop - start
    logger.debug('Prediction completed : {} : {}'.format(cal_time, 'damage-model'))
    inference_time = round(cal_time, 2)

    # Post-Process
    # Some slow numpy postprocessing:
    boxes = boxes / scale
    # boxes are already clipped inside the graph, but after the floating point scaling, this may not be true any more.
    boxes = clip_boxes(boxes, orig_shape)
    if masks:
        full_masks = [_paste_mask(box, mask, orig_shape)
                      for box, mask in zip(boxes, masks[0])]
        masks = full_masks
    else:
        # fill with none
        masks = [None] * len(boxes)

    results = [DetectionResult(*args) for args in zip(boxes, probs, labels.tolist(), masks) if args[1] > thresh]

    # Reporting
    ## Make report
    if len(results) > 0:
        boxes = np.asarray([r.box for r in results])
        probs = np.asarray([r.score for r in results])
        labels = np.asarray([r.class_id for r in results])

        areas = np_area(boxes)
        _rule_minsize = areas < 0  # TEMP
        sorted_inds = np.argsort(-areas)
        report_set = {"class": [],
                      "class_name": [],
                      "box": [],
                      "area": [],
                      "score": [],
                      "rule_minsize": []
                      }

        report = dict()
        for idx, ids in enumerate(sorted_inds):
            _class = labels[ids]
            if _class == 0: continue  # BG pass
            report[str(idx)] = copy.deepcopy(report_set)

            report[str(idx)]["class"] = _class
            report[str(idx)]["class_name"] = DAMAGE_DF.loc[DAMAGE_DF["class"] == _class].index[0]
            report[str(idx)]["box"] = boxes[ids]
            report[str(idx)]["area"] = areas[ids]
            report[str(idx)]["score"] = probs[ids]
            report[str(idx)]["rule_minsize"] = _rule_minsize[ids]

        ## Make segmap
        class_list = np.setdiff1d(DAMAGE_DF.index.to_numpy(), "BG")
        class_id_list = DAMAGE_DF.loc[DAMAGE_DF.index.to_series().apply(lambda x: x in class_list), "class"].values
        class_intersect_idx = np.intersect1d(class_id_list, labels)
        class_intersect = DAMAGE_DF.index[class_intersect_idx]
        seg_map_canvas = np.zeros(orig_shape, dtype=np.int)

        seg_map = dict()
        for elem in report.values():
            _class_name = elem["class_name"]
            _class = elem["class"]
            _box = np.round(elem["box"]).astype(np.int)
            _seg_map = copy.deepcopy(seg_map_canvas)

            if _class_name not in seg_map.keys():
                seg_map[_class_name] = []

            _seg_map[_box[1]:_box[3], _box[0]:_box[2]] = _class
            seg_map[_class_name].append(np.expand_dims(_seg_map, 0))

        ## Summary
        for_summary_df = DAMAGE_DF.loc[class_intersect].loc[(DAMAGE_DF["for_dnn"] == True)].sort_values("order", ascending=True)

        seg_map_summary = []
        for _class_name in for_summary_df["class_name"].values:
            _stacked_seg_map = np.vstack(seg_map[_class_name])
            _class_seg_map = np.max(_stacked_seg_map, axis=0, keepdims=True)

            seg_map[_class_name] = np.vstack([_stacked_seg_map, _class_seg_map])

            if len(np.intersect1d(np.unique(_class_seg_map), for_summary_df["class"].values)) > 0:
                seg_map_summary.append(_class_seg_map)

        if len(seg_map_summary) > 0:
            _stacked_seg_map = np.vstack(seg_map_summary)
            _summary_seg_map = np.vstack([_stacked_seg_map, \
                                          np.max(_stacked_seg_map, axis=0, keepdims=True)])
        else:
            _summary_seg_map = np.expand_dims(copy.deepcopy(seg_map_canvas), 0)

        seg_map["summary"] = _summary_seg_map

    else:
        report = dict()
        seg_map = dict()

    # print(hm.check('[{}] After damage predict'.format(threading.get_ident())), flush=True)

    return report, seg_map, inference_time


def parts_inference_trtserv(model_name, input_image, url=None):
    if url is None:
        url = config['PARTS_SVC']
    protocol = ProtocolType.from_str('http')
    streaming = False
    verbose = False
    model_version = -1
    batch_size = 1

    #################################################################
    ## preprocess
    gen_input = gen_inputs(config=MASK_CONFIG)

    # preprocess
    molded_images, image_metas, windows, anchors = \
        gen_input.preprocess(input_image)

    #################################################################
    ## call inference serving
    for iter in range(5):
        try:
            infer_ctx = InferContext(url, protocol, model_name,
                                     model_version, verbose=verbose, streaming=streaming)

            result = infer_ctx.run({'input_image': (molded_images,),
                                    'input_image_meta': (image_metas,),
                                    'input_anchors': (anchors,)},
                                   {'mrcnn_detection/Reshape_1': InferContext.ResultFormat.RAW,
                                    'mrcnn_mask/Reshape_1': InferContext.ResultFormat.RAW},
                                   batch_size)

            detections = result['mrcnn_detection/Reshape_1'][0]
            mrcnn_mask = result['mrcnn_mask/Reshape_1'][0]

            break
        except InferenceServerException as infer_e:
            if iter < 4:
                logger.error('Inference (%s) exception. retry to infer.' % model_name)
                time.sleep(0.2)
            else:
                logger.error('Inference({}|{}) exception. retry FAILED. {}'.format(url,
                                                                                   model_name,
                                                                                   infer_e))
                raise infer_e
        finally:
            if infer_ctx:
                infer_ctx.close()
    #################################################################
    # Postprocess

    # Process detections
    results = []
    for i, image in enumerate(molded_images):
        final_rois, final_class_ids, final_scores, final_masks = \
            gen_input.unmold_detections(detections=detections[i],
                                        mrcnn_mask=mrcnn_mask[i],
                                        original_image_shape=image.shape,
                                        image_shape=molded_images[i].shape,
                                        window=windows[i])

        results.append({"rois": final_rois,
                        "class_ids": final_class_ids,
                        "scores": final_scores,
                        "masks": final_masks
                        })

    return results


def damage_inference_trtserv(model_name, input_image, url=None):
    # url = 'tensorrt-damage-modelservice.aosprod.svc.cluster.local:8000'
    # url = 'tensorrt-damage-modelservice:8000'
    if url is None:
        url = config['DAMAGE_SVC']
    # url = 'ingressgateway:80'
    protocol = ProtocolType.from_str('http')
    streaming = False
    verbose = False
    model_version = -1
    batch_size = 1

    #################################################################
    ## preprocess
    image = input_image.astype(np.float32)

    #################################################################
    ## call inference serving
    for iter in range(5):
        try:
            infer_ctx = InferContext(url, protocol, model_name,
                                     model_version, verbose=verbose, streaming=streaming)

            result = infer_ctx.run({'image': (image,)},
                                   {'output/boxes': (InferContext.ResultFormat.RAW),
                                    'output/scores': (InferContext.ResultFormat.RAW),
                                    'output/labels': (InferContext.ResultFormat.RAW)},
                                   batch_size)
            break
        except InferenceServerException as infer_e:
            if iter < 4:
                logger.error('Inference (%s) exception. retry to infer.' % model_name)
                time.sleep(0.2)
            else:
                logger.error('Inference({}|{}) exception. retry FAILED. {}'.format(url,
                                                                                   model_name,
                                                                                   infer_e))
                raise infer_e
        finally:
            if infer_ctx:
                infer_ctx.close()
                del (infer_ctx)

    #################################################################
    # Postprocess

    boxes = result['output/boxes'][0]
    labels = result['output/labels'][0]
    scores = result['output/scores'][0]

    return boxes, scores, labels


def inception_inference_trtserv(url, model_name, input_image, filter=False):
    # protocol = ProtocolType.from_str('http')
    protocol = ProtocolType.from_str('http')
    streaming = False
    verbose = False
    model_version = -1
    batch_size = 1

    #################################################################
    ## preprocess
    # Get batch size
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, 0)
    num_image = len(input_image)

    if config['MODEL_BATCH'] == 'true':
        input_images = list()
        for idx in range(batch_size):
            input_images.append(input_image[0])

    output_name = 'output/Softmax'

    #################################################################
    ## call inference serving
    infer_ctx = None
    for iter in range(5):
        try:
            # logger.info('call filter modelservice. {}|{}'.format(url, model_name))
            infer_ctx = InferContext(url, protocol, model_name,
                                     model_version, verbose=verbose, streaming=streaming)
            if filter:
                if config['MODEL_BATCH'] == 'true':
                    result = infer_ctx.run({'densenet169_input': input_images},
                                           {output_name: (InferContext.ResultFormat.RAW)},
                                           batch_size)
                else:
                    result = infer_ctx.run({'densenet169_input': (input_image,)},
                                           {output_name: (InferContext.ResultFormat.RAW)},
                                           batch_size)
            # elif 'lamp' in model_name:
            #     if config['MODEL_BATCH'] == 'true':
            #         result = infer_ctx.run({'inception_v4_input': input_images},
            #                                {'output/Softmax': (InferContext.ResultFormat.RAW)},
            #                                batch_size)
            #     else:
            #         result = infer_ctx.run({'inception_v4_input': (input_image,)},
            #                                {'output/Softmax': (InferContext.ResultFormat.RAW)},
            #                                batch_size)
            else:
                output_name = 'classifier_head/head_softmax/Softmax'
                if 'v0' in model_name:
                    output_name = 'output/Softmax'

                if config['MODEL_BATCH'] == 'true':
                    result = infer_ctx.run({'efficientnet-b5_input': input_images},
                                           {output_name: (InferContext.ResultFormat.RAW)},
                                           batch_size)
                else:
                    result = infer_ctx.run({'efficientnet-b5_input': (input_image,)},
                                           {output_name: (InferContext.ResultFormat.RAW)},
                                           batch_size)
            break
        except InferenceServerException as infer_e:
            if iter < 4:
                logger.error('Inference({}) exception. retry to infer.[{}]'.format(model_name, iter))
                time.sleep(0.2)
            else:
                logger.error('Inference({}-{}) exception. retry FAILED. {}'.format(url,
                                                                                   model_name,
                                                                                   infer_e))
                raise infer_e
        finally:
            if infer_ctx:
                infer_ctx.close()
                del (infer_ctx)

    return result[output_name][0]
