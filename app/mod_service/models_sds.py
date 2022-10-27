import os, json, time
from pprint import pprint
from collections import namedtuple
import matplotlib.pyplot as plt

from app.config import config

################ Denoise ##############
from concurrent import futures
import grpc

import app.mod_service.sds_grpc.sds_removal_pb2 as sds_removal_pb2
import app.mod_service.sds_grpc.sds_removal_pb2_grpc as sds_removal_pb2_grpc
from io import BytesIO

import numpy as np
import cv2
################ Denoise ##############
################ Damage ##############
from app.mod_service.sds_grpc import sds_damage_segmentation_pb2
from app.mod_service.sds_grpc import sds_damage_segmentation_pb2_grpc
from skimage.transform import resize
from skimage.measure import label, regionprops
################ Damage ##############


############### Image Matching ########
from app.mod_service.sds_grpc import sds_image_matching_pb2
from app.mod_service.sds_grpc import sds_image_matching_pb2_grpc

from app.ai.utils import *
from app.ai.tools_sds import *
from app.ai.config import *
from app.ai.fasterrcnn.common import CustomResize, clip_boxes
from app.ai.fasterrcnn.eval import _paste_mask
from app.common.model_service import SodaModelService

from app.utility.custom_utils import create_json, draw_result, get_files, save_json, make_polygon , get_bbox_and_polygon, recoord, polygon_combine_area, seg_polygon
from app.utility.custom_utils import decode_mask_to_rle
from shapely.ops import cascaded_union, unary_union
from shapely.geometry import Point, Polygon
import glob

# from tensorrtserver.api import ProtocolType, InferContext, InferenceServerException

logging.getLogger('tensorflow').setLevel(logging.ERROR)
logger = create_logger(__name__)



################ service call #################################################

def model_service_meta(param):
    logger.info('model service meta called.')

    model_service = SodaModelService(
        server_url=param['svc_name'],
        model_name=param['model_name'],
        # model_version=param['model_version']
    )
    logger.debug(f"Call model service meta info . {param['svc_name']}:{param['model_name']}")
    meta = model_service.get_model_meta_data()
    return meta

def call_model_service(server_url, model_name, input_img):
    param = {
        'svc_name': server_url,
        'model_name': model_name,
        # 'model_version': model_version
    }
    meta_data = model_service_meta(param)

    input_meta = meta_data["inputs"][0]

    inputs = [{
        "name":input_meta["name"],
        "shape":input_img.shape, 
        "dtype":input_meta["datatype"],
        "input_tensor":input_img
        }]

    output_meta = meta_data["outputs"]
    outputs = []
    for om in output_meta:
        od = {"name":om["name"]}
        outputs.append(od)

    model_service = SodaModelService(
        server_url=param['svc_name'],
        model_name=param['model_name'],
        # model_version=param['model_version']
    )

    model_service.add_inputs(inputs)
    model_service.add_outputs(outputs)

    logger.info(f"Call model service prediction . {param['svc_name']}:{param['model_name']}")

    results = model_service.predict()

    return results

#################################################################


######### filter model ####################################################

def filter_cls_detection(data, image_out=True):
    inference_time = 0

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
    input_image = cv2.resize(image, (shape, shape))
    input_image = np.expand_dims(input_image, 0).astype(np.float32)
    if np.nanmax(input_image) > 1 :
        input_image = input_image/255.

    # Check orc, dashboard, vincode
    if mobile_class in fast_pass_list:
        probs = 0
        score = 0
        model_cls = FILTER_DF.loc[FILTER_DF.index == mobile_class, "class"].values[0]
        filter_code = filter_class_to_code(model_cls=model_cls)
    else:
        start = time.time()
        model_name = '{}'.format(config['FILTER_MODEL'])
        url = config['FILTER_SVC']

        # probs = inception_inference_trtserv(url, model_name, input_image, True)
        probs = call_filter_modelsvc(url, model_name, input_image, True)
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
    print("## filter_cls_detection Inferecne_time : {}".format(inference_time))
    return report, inference_time

def call_filter_modelsvc(svc_name, model_name, input_image, filter=False):
    batch_size = 1

    #################################################################
    # preprocess

    # Get batch size
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, 0)
    num_image = len(input_image)

    # if config['MODEL_BATCH'] == 'true':
    #     input_images = list()
    #     for idx in range(batch_size):
    #         input_images.append(input_image[0])

    #################################################################
    ## call inference serving
    filter_service_result = call_model_service(
        server_url='{}'.format(config['FILTER_SVC']),
        model_name='{}'.format(config['FILTER_MODEL']),
        # model_version= '{}'.format(config['FILTER_MODEL_VERSION']),
        input_img=np.array(input_image)
    )
    output_softmax = filter_service_result["OUTPUT_SOFTMAX"][0]
    output_class = filter_service_result["OUTPUT_CLASS"][0]
    #################################################################

    return output_softmax

#####################################################################




######### parts model ####################################################

def parts_detection(data, model_name=None, model_svc=None, masking=False, masking_output_path=None):
    inference_time = 0
    # logger.debug('parts detection serve')
    image_data = data['data_res_out']
    image_name = data["data_res_out"]["url"].split('/')[-1]

    # Pre-Processing
    input_image = [resize_image(image_data['image'],
                                min_dim=MASK_CONFIG.IMAGE_MIN_DIM,
                                max_dim=MASK_CONFIG.IMAGE_MAX_DIM,
                                min_scale=0)[0]]

    # Inference
    start = time.time()
    part_res_out = call_parts_modelsvc(input_image, image_name)
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
        thresh_idx = [index for index,value in enumerate(part_res_out['scores']) if value > thresh]
        # thresh_idx = part_res_out['scores'] >= thresh
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
        #image_masked = input_image[0].copy()
        #image_masked[np.where(mask_expanded == False)] = np.array([0, 0, 0], dtype=np.uint8)

        # Check rule minimun-size
        _rule_minsize = False

        # Check rule close
        _rule_close = rule_close(resized_mask)

        # Check rule focus (_rule_focu : #true,false,
        _rule_focus, inbox_ratio = rule_focus(resized_mask)
        #logger.info('rule_focus = {}, inbox_ratio = {}'.format(_rule_focus, inbox_ratio))
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
        #res_out["image_masked"]  = [image_masked]
        res_out["inbox_ratio"]   = [inbox_ratio]
        res_out["rule_minsize"]  = [_rule_minsize]
        res_out["rule_focus"]    = [_rule_focus]
        res_out["rule_close"]    = [_rule_close]
        res_out["rule_dup"]      = [_rule_dup]
        res_out["rule_cont"]     = [_rule_cont]
        res_out["rule_out"]      = [_rule_out]

        output_list.append(pd.DataFrame(res_out))

    # Make output df # try except 문 추가 (2021-04-15) LKS
    try:
        output_df = pd.concat(output_list).reset_index(drop=True)
    except:
        output_df = []
        report = {"check": False,
                  "report": output_df
                  }
        return report, inference_time

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

            polygon_list=[]
            part_name_list=[]
            score_list=[]
            for part_name in part_res.keys():
                if part_res[part_name]["score"] >= thresh:
                    mask_for_tool[np.where(part_res[part_name]["resized_mask"] == True)] = np.array(
                        PARTS_DF.loc[part_name, "rgb"], dtype=np.uint8)
                    
                    part_name_list.append(part_name)
                    score_list.append(part_res[part_name]["score"])
                    polygon_point=make_polygon(part_res[part_name]["resized_mask"])
                    polygon_list.append(polygon_point)
                    part_res[part_name]["polygon_point"] = polygon_point


            result_image = draw_result(
                PARTS_DF["part_name"],
                image_data['image'],
                polygon_list,
                part_name_list,
                score_list,
                out_file=None,
                color_list=None,
            )
            cv2.imwrite('./output/{0}'.format('polygon_'+image_name),result_image)

            alpa=0.7
            add_img = cv2.addWeighted(image_data['image'], alpa, mask_for_tool, (1-alpa), 0) 
            cv2.imwrite( './output/{0}'.format('mask_'+image_name), add_img)
            
            report = {"check": len(part_res) > 0,
                      "mask_output": masking_output_path,
                      "report": part_res
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
    stop = time.time()
    cal_time2 = round(stop - start,2)

    print("## parts_detection Inferecne_time : {}".format(inference_time))
    return report, inference_time

def call_parts_modelsvc(input_image, image_name):
    image_name = image_name.split('.')[0]
    
    batch_size = 1

    # #################################################################
    # ## preprocess
    # gen_input = gen_inputs(config=MASK_CONFIG)

    # # preprocess
    # molded_images, image_metas, windows, anchors = \
    #     gen_input.preprocess(input_image)

    #################################################################


    #################################################################
    ## call inference serving
    part_service_result = call_model_service(
        server_url='{}'.format(config['PARTS_SVC']),
        model_name='{}'.format(config['PARTS_MODEL']),
        input_img=np.array(input_image)
    )

    #################################################################
    # Postprocess
    results = {
        "rois": np.array(part_service_result["OUTPUT_BBOX"]) if part_service_result["OUTPUT_BBOX"]!=None else None,
        "class_ids": np.array(part_service_result["OUTPUT_CLS_IDS"]) if part_service_result["OUTPUT_CLS_IDS"]!=None else None,
        "scores": np.array(part_service_result["OUTPUT_SCORE"]) if part_service_result["OUTPUT_SCORE"]!=None else None,
        "sizes": np.array(part_service_result["OUTPUT_MASK_SIZE"]) if part_service_result["OUTPUT_MASK_SIZE"]!=None else None,
        "masks": part_service_result["OUTPUT_MASK_RLE"] if part_service_result["OUTPUT_MASK_RLE"]!=None else None
    }
    # print(image_name, np.array(part_service_result["OUTPUT_BBOX"]), np.array(part_service_result["OUTPUT_SCORE"]),
    # np.array(part_service_result["OUTPUT_MASK"]).shape, np.array(part_service_result["OUTPUT_CLS_IDS"]))

    masksize = results["sizes"]
    maskrles = results["masks"]
    masks = decode_mask_to_rle(masksize, maskrles)

    # end = time.time()
    # print("time", end-start)

    results["masks"] = masks
    del results["sizes"]

    return results

##################################################################


################ Damage ##############
DATA_CLASSES = ["scratch", "dent", "complex_damage", "broken"]
SCORE_THR = 0.3

def damage_detection(data, model_name=None, model_svc=None, closed_shot=False):
    inference_time = 0
    # DetectionResult = namedtuple(
    #     'DetectionResult',
    #     ['box', 'score', 'class_id', 'mask'])

    image = data["data_res_out"]["image"]
    orig_shape = data["data_res_out"]["shape"][:2]
    thresh = model_option["damage"]["score_thres"]
    image_path = data["data_res_out"]["url"]

    #################################################################
    ## Get output
    start = time.time()
    if model_name is None:
        model_name = '{}'.format(config['DAMAGE_MODEL'])

    # inference service call
    report = damage_send(image, image_path, closed_shot)
    stop = time.time()
    cal_time = stop - start
    logger.debug('Prediction completed : {} : {}'.format(cal_time, 'damage-model'))
    inference_time = round(cal_time, 2)
    #################################################################

    print("## damage_detection Inferecne_time : {}".format(inference_time))
    return report, inference_time


def damage_send(imgdata, image_path, closed_shot=False):
    '''
    meta
    {
        'name': 'twincar-damage-model',
        'inputs': [{'datatype': 'UINT8','name': 'INPUT__0','shape': ['-1', '-1', '-1', '-1']}],
        'outputs': [{'datatype': 'BYTES','name': 'OUTPUT__0','shape': ['-1', '-1', '-1']}],
        'platform': 'python',
        'versions': ['2']}
    }
    '''

    ## resize and normalize image
    img_resize = (401, 401)  
    imgdata = cv2.resize(imgdata, dsize = img_resize)

    ## inference service call
    bbox_result = call_model_service(
        server_url='{}'.format(config['DAMAGE_SVC']),
        model_name='{}'.format(config['DAMAGE_MODEL']),
        input_img=np.array([imgdata]),
    )
    bbox_result = bbox_result["OUTPUT__0"]

    ## data fomatting / image save
    # damage_result = damage_image_analysis(bbox_result, imgdata, img_path=image_path)
    damage_result = damage_image_analysis(bbox_result, imgdata)
    ''' 
    damage_result ={
        "damage_name":classes,
        "scores":scores,
        "polygons":polygons,
    }
    '''

    ## damage 여부 식별
    if len(damage_result["damage_name"]) >0 :
        return_data ={
            "image": imgdata,
            "damage_state": True,
            "damage_result":damage_result,
        }
    else :
        return_data ={
            "image": imgdata,
            "damage_state": False,
        }

    return return_data

def damage_image_analysis(result, img_data, img_path=None):

    ## damage class 별 score, polygon 추출 및 이미지 저장
    classes = []
    scores = []
    polygons = []

    # create result
    for bbox in result:
        score = round(float(bbox[3]), 2)
        if score < SCORE_THR:
            continue
        scores.append(score)
        classes.append(DATA_CLASSES[int(bbox[0])])
        poly = [float(x) for x in bbox[1].decode().split(",")]
        polygons.append([[poly[2 * i], poly[2 * i + 1]] for i in range(4)])
        [int(x) for x in bbox[2].decode().split(",")]

    if img_path:
        # image save process 
        result_output_dir='{}'.format(config['DAMAGE_IMAGE_PATH'])
        input_image_data = np.array(img_data)

        json_dir = os.path.join(result_output_dir, "json")
        os.makedirs(json_dir, exist_ok=True)
        img_dir = os.path.join(result_output_dir, "imgs")
        os.makedirs(img_dir, exist_ok=True)

        basename = os.path.basename(img_path)
        exts = basename.split(".")[-1]
        unique_name = basename.replace(f".{exts}", "")

        json_save_path = os.path.join(json_dir, unique_name + ".json")
        img_save_path = os.path.join(img_dir, unique_name + ".jpg")

        # image save result
        result_image = draw_result(
            DATA_CLASSES,
            input_image_data,
            polygons,
            classes,
            scores,
            out_file=None,
            color_list=None,
        )
        cv2.imwrite(img_save_path, result_image)
    
    # output format
    damage_result ={
        "damage_name":classes,
        "scores":scores,
        "polygons":polygons,
    }

    return damage_result

#################################################################


# damage_send_opt = [
#     ('grpc.max_send_message_length', 20 * 1024 * 1024),
#     ('grpc.max_receive_message_length', 20 * 1024 * 1024)
# ]
# damage_svc = config['SDS_DAMAGE_SVC']
# close_damage_svc = config['SDS_DAMAGE_CLOSE_SVC']
# damage_channel = grpc.insecure_channel(f"{damage_svc}", options=damage_send_opt)
# close_damage_channel = grpc.insecure_channel(f"{close_damage_svc}", options=damage_send_opt)
# damage_stub = sds_damage_segmentation_pb2_grpc.GreeterStub(damage_channel)
# close_damage_stub = sds_damage_segmentation_pb2_grpc.GreeterStub(close_damage_channel)
# def damage_send(imgdata, closed_shot=False):
#     def ndarray_to_proto(nda):
#         nda_bytes = BytesIO()
#         np.save(nda_bytes, nda, allow_pickle=False)
#         return sds_damage_segmentation_pb2.NDArray(ndarray=nda_bytes.getvalue())

#     def proto_to_ndarray(nda_proto):
#         nda_bytes = BytesIO(nda_proto.ndarray)
#         return np.load(nda_bytes, allow_pickle=False)

#     # url = '192.168.226.26'
#     # port = '50051'
#     stub = damage_stub
#     if closed_shot:
#         # port = '50052'
#         stub = close_damage_stub
#     # with grpc.insecure_channel(f'{url}:{port}', options=opt) as channel:
#     logger.debug(f"Call Damage model service : [{damage_svc}]")
    
#     ## resize and normalize image
    
#     img_resize = (401, 401)  
#     imgdata = cv2.resize(imgdata, dsize = img_resize)
#     imgdata = imgdata/255.
    
#     means = np.array([0.485, 0.456, 0.406])
#     stds = np.array([0.229, 0.224, 0.225])
#     imgdata = (imgdata-means) / stds
    
#     imgdata = np.transpose(imgdata, (2,0,1))
#     imgdata = np.expand_dims(imgdata, 0)
    
#     tempimgdata = ndarray_to_proto(imgdata)

#     response = stub.inference(tempimgdata)
#     response_temp = proto_to_ndarray(response)
#     response = response_temp[0]
        
#     return response