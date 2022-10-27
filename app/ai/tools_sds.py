from app.ai.config import *
from app.config import config

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import os
import copy
import json
import io
import requests
json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

from decimal import Decimal
from datetime import datetime

from app.utility.custom_utils import *
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

# Common
class time_checker :
    def __init__(self) :
        self.last_point = datetime.now()
    
    def point(self) :
        self.last_point = datetime.now()
        
    def check(self) :
        temp_time = datetime.now()
        self.duration = temp_time - self.last_point
        self.last_point = temp_time
        
        return round(self.duration.total_seconds(), 2)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return json.JSONEncoder.default(self, obj)

def jsonify(data):
    json_data = dict()
    for key, value in data.items():
        if isinstance(value, list): # for lists
            value = [ jsonify(item) if isinstance(item, dict) else item for item in value ]
        if isinstance(value, dict): # for nested lists
            value = jsonify(value)
        if isinstance(key, int): # if key is integer: > to string
            key = str(key)
        if type(value).__module__=='numpy': # if value is numpy.*: > to python list
            value = value.tolist()
        json_data[key] = value
    return json_data


def get_image(data):
    try:
        image = None
        check = True

        # pprint(data)

        if type(data) is str:
            url = data
            image = cv2.imread(url, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            url = data['image_url']
            data = data.get('image_data', None)
            if data:
                # response = requests.get(url)
                # content = BytesIO(response.content)
                # image = np.array(Image.open(content))
                image = np.array(Image.open(data))
            elif os.path.isfile(url):
                image = cv2.imread(url, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                key = os.path.basename(url)
            else:
                response = requests.get(url)
                content = io.BytesIO(response.content)
                image = np.array(Image.open(content))
                # check = False
    except Exception as e:
        print('exception -> %s' % e)
        image = None
        check = False
    
    return {"check": check, "url": url, "image": image, "shape": image.shape if image is not None else None}

def show_im(img, size=12) :
    if type(img) == str :
        img = plt.imread(img)

    plt.figure(figsize=(size, size))
    plt.imshow(img)
    plt.show()

def mobile_info_parser(url) :
    mobile_info = {"check": False,
                   "class": None,
                   "mode" : None,
                   "tiny" : None,
                   "part_code" : None,
                   "part_name" : None
                  }

    if isinstance(url, str) :

        if "ocr_" in url :
            mobile_info["class"] = "ocr"

        elif "vin_" in url :
            mobile_info["class"] = "vin"

        elif "dash_" in url :
            mobile_info["class"] = "dash"

        elif "parts_" in url :
            mobile_info["class"] = "parts"

            # mode
            if "_ai_" in url :
                mobile_info["mode"] = "ai"
            elif "_nor_" in url :
                mobile_info["mode"] = "nor"

            # tiny
            if "_tiny_" in url :
                mobile_info["tiny"] = True
            else :
                mobile_info["tiny"] = False

            # part_name
            if "_DGPT" in url :
                part_code = "DGPT" + url.split("_DGPT")[-1].split("_")[0]
                if part_code == "DGPT99":
                    part_name = "mobile_etc"
                else:
                    part_name = PARTS_DF.loc[PARTS_DF["part_code"] == part_code, "part_name"].values[0]

                mobile_info["part_code"] = part_code
                mobile_info["part_name"] = part_name

        elif "nor_" in url :
            mobile_info["class"] = "nor"

        elif "fix_" in url :
            mobile_info["class"] = "fix"

        checker = np.array(list(mobile_info.values()))
        if not np.all(checker[1:] == None) :
            mobile_info["check"] = True

    return mobile_info


def aos_info_parser(part_code, is_group):
    aos_info = {"check": False,
                "part_code": None,
                "part_name" : None,
                "part_group": None
                }

    if isinstance(part_code, str) :
        if "DGPT" in part_code:
            aos_info["part_code"] = part_code

            all_part_list = PARTS_DF["part_code"].values
            if part_code in all_part_list:
                part_name = PARTS_DF.loc[PARTS_DF["part_code"] == part_code, "part_name"].values[0]

                aos_info["part_name"] = part_name
                aos_info["part_group"] = is_group

            elif part_code == "DGPTXX" :
                aos_info["part_name"] = "AOS_EXCLUDE"
                aos_info["part_group"] = False

        if aos_info["part_name"] is not None:
            aos_info["check"] = True

    return aos_info


def remove_padding(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image

def crop_or_pad(image, size=(299, 299)) :
    crop_image = None
        
    crop_image = remove_padding(image)
    crop_image = crop_image.astype(np.float32)
    h, w = crop_image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    image_max = max(h, w)
    scale = float(min(size)) / image_max

    crop_image = cv2.resize(crop_image, (int(w * scale), int(h * scale)))

    h, w = crop_image.shape[:2]
    top_pad = (size[1] - h) // 2
    bottom_pad = size[1] - h - top_pad
    left_pad = (size[0] - w) // 2
    right_pad = size[0] - w - left_pad
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    crop_image = np.pad(crop_image, padding, mode='constant', constant_values=0)

    # Fix image normalization
    if np.nanmax(crop_image) > 1 :
        crop_image = np.divide(crop_image, 255.)

    return crop_image

def check_valid(data, key=None) :
    res = ""
    try :
        if (type(data) == dict) & (key is not None) :
            if key in data.keys() :
                res = data[key]
            else :
                res = None
        else :
            res = data
            
        if type(res) == np.ndarray :

            if len(res.shape) == 1 :
                res = res.tolist()
            else :
                res = res

        if key in ["probs", "score"] :
            if type(res) == list :
                res = [round(x, 4) for x in res]
            elif type(res) == float :
                res = round(res, 4)
            
    except Exception as _except :
        print("EXCEPTION # check_valid :", _except)
        res = None
                
    return res

def repair_class_to_code(probs, model_cls=None) :
    n_class = len(probs)
    
    if model_cls is None :
        model_cls = np.argmax(probs)
    
    if n_class == 3 :
        code_table = ["RPCL00", "RPCL21", "RPCL99"]

    else :
        code_table = ["RPCL00", "RPCL12", "RPCL31", "RPCL32", "RPCL33", "RPCL99"]
        code_table = code_table[:(n_class-1)] + ["RPCL99"]

    repair_code = code_table[model_cls]

    return repair_code

def filter_class_to_code(model_cls) :
    return "AIFT0"+str(model_cls)

def is_in_frame(mask, thresh=0.4) :
    def get_mask_length(mask, axis) :
        index_array = np.where(np.any(mask, axis=axis) == True)[0]

        if len(index_array) >= 2 :
            length = np.nanmax(index_array) - np.nanmin(index_array)
        else :
            length = 0

        return length

    mask_height = get_mask_length(mask=mask, axis=1)
    mask_width = get_mask_length(mask=mask, axis=0)

    # division zero handle
    if np.sum(np.array([mask_height, mask_width]) == 0) > 0:
        checker = np.array([999]*4)

    else:
        image_height, image_width = mask.shape
        image_max_len = np.nanmax([image_height, image_width])
        pixel=np.clip(int(image_max_len * 0.05), 3, 10)

        checker = []
        # check left
        checker.append(get_mask_length(mask[:, :pixel], axis=1) / mask_height)
        # check rigth
        checker.append(get_mask_length(mask[:, image_width-pixel:], axis=1) / mask_height)
        # check up-side
        checker.append(get_mask_length(mask[:pixel, :], axis=0) / mask_width)
        # check bottom
        checker.append(get_mask_length(mask[image_height-pixel:, :], axis=0) / mask_width)

        checker = np.array(checker)

    return np.nansum(checker < thresh) # if true : in frame // return : in frame counts


def apply_mask(image, seg_image, alpha=0.3):
    _image = copy.deepcopy(image)
    mask_area = np.squeeze(np.max(seg_image, axis=-1, keepdims=True))
    
    _image = np.where(np.dstack([mask_area] * 3) > 0,
                      _image * (1 - alpha) +
                      alpha * seg_image,
                      _image).astype(np.uint8)

    return _image


def segmap2image(seg_map, df, df_var="class") :
    image = np.zeros(list(seg_map.shape) + [3], dtype=np.uint8)
    for model_cls in [x for x in np.unique(seg_map) if x != 0] :
        image[np.where(seg_map == model_cls)] = np.array(df.loc[df[df_var] == model_cls]["rgb"].values[0], dtype=np.uint8)

    return image

def get_damaged_section(data, part_name) :
    parts_damage_dict = data["part_res_out"]["report"][part_name]

    parts_mask = parts_damage_dict["resized_mask"]
    if "complex_damage" in parts_damage_dict["damage"].keys() :
        _dam_cls = DAMAGE_DF.loc[DAMAGE_DF["class_name"] == "complex_damage", "class"].values[0]
        parts_damage_seg_map = parts_damage_dict["damage"]["complex_damage"]["seg_map"]
        parts_damage_seg_map = parts_damage_seg_map == _dam_cls
    else :
        parts_damage_seg_map = np.zeros(parts_mask.shape, dtype=np.bool)

    damaged_area = np.sum(parts_damage_seg_map)

    section_len = PARTS_DF.loc[part_name, "section_len"]
    ref_part_list = PARTS_DF.loc[part_name, "section"]

    check_list = {"left"  : [],
                  "right" : [],
                  "center": [],
                  "damage_loc" : []
                 }

    # Make loc name
    ref_point = {"section_name" : [],
                 "is_damaged" : [],
                 "points" : [],
                 "x_ref" : [],
                 "y_ref" : []
                }
    
    if ref_part_list is None :
        ref_part_list = []
        
    intersect_list = np.intersect1d(ref_part_list, 
                                    list(data["part_res_out"]["report"].keys()))
    
    # Check left
    check_list["left"] = [x for x in intersect_list if "left" in x]
    # Check right
    check_list["right"] = [x for x in intersect_list if "right" in x]
    # Check center
    check_list["center"] = [x for x in intersect_list if x in ["rear_bumper"]]

    if (part_name in ["front_bumper", "rear_bumper"]) & (len(intersect_list) > 0) :
        # Get reference point
        for inter_parts in check_list["left"][:1] + check_list["right"][:1] :
            inter_parts_mask = data["part_res_out"]["report"][inter_parts]["resized_mask"]
            index_array = np.where(np.any(inter_parts_mask, axis = 0) == True)[0]
            ref_point["points"].extend([np.nanmin(index_array), np.nanmax(index_array)])

        # Check damaged
        if len(ref_point["points"]) == 4 :
            ref_point["points"] = np.sort(ref_point["points"])[np.array([1, 2])]
            ref_point["x_ref"] = [np.nanmin(ref_point["points"]), np.nanmax(ref_point["points"])]

            ref_point["is_damaged"].append(np.sum(parts_damage_seg_map[:, :ref_point["x_ref"][0]]) / damaged_area)
            ref_point["is_damaged"].append(np.sum(parts_damage_seg_map[:, ref_point["x_ref"][0]:ref_point["x_ref"][1]]) / damaged_area)
            ref_point["is_damaged"].append(np.sum(parts_damage_seg_map[:, ref_point["x_ref"][1]:]) / damaged_area)

            ref_point["section_name"] = ["DGRGL", "DGRGC", "DGRGR"]

            if part_name == "front_bumper" :
                ref_point["section_name"] = list(reversed(ref_point["section_name"]))

        elif len(ref_point["points"]) == 2 :

            if len(check_list["left"][:1]) > 0 :
                ref_point["section_name"] = ["DGRGL", "DGRGC"]
            elif len(check_list["right"][:1]) > 0 :
                ref_point["section_name"] = ["DGRGC", "DGRGR"]

            idx_point = np.where(np.array(ref_point["section_name"]) == "DGRGC")[0][0]

            if part_name == "front_bumper" :
                idx_point = np.abs(idx_point-1)
                ref_point["section_name"] = list(reversed(ref_point["section_name"]))

            ref_point["points"] = ref_point["points"][idx_point]
            ref_point["x_ref"] = [ref_point["points"]]

            ref_point["is_damaged"].append(np.sum(parts_damage_seg_map[:, :ref_point["x_ref"][0]]) / damaged_area)
            ref_point["is_damaged"].append(np.sum(parts_damage_seg_map[:, ref_point["x_ref"][0]:]) / damaged_area)

    if (part_name in ["back_door", "trunk"]) & (len(intersect_list) > 0) :
        ref_point["section_name"] = ["DGRGL", "DGRGR"]

        # Get reference point
        if (len(check_list["left"]) > 0) & (len(check_list["right"]) > 0) :
            for inter_parts in check_list["left"][:1] + check_list["right"][:1] :
                inter_parts_mask = data["part_res_out"]["report"][inter_parts]["resized_mask"]
                index_array = np.where(np.any(inter_parts_mask, axis = 0) == True)[0]
                ref_point["points"].append([np.nanmin(index_array), np.nanmax(index_array)])

            ref_point["x_ref"] = [ref_point["points"][0][1] + (ref_point["points"][1][0] - ref_point["points"][0][1])//2]

        elif is_in_frame(parts_mask) == 4 :
            index_array = np.where(np.any(parts_mask, axis = 0) == True)[0]
            ref_point["points"].append([np.nanmin(index_array), np.nanmax(index_array)])
            ref_point["x_ref"] = [ref_point["points"][0][0] + (ref_point["points"][0][1] - ref_point["points"][0][0])//2]

        # Check damaged
        if len(ref_point["x_ref"]) > 0 :
            ref_point["is_damaged"].append(np.sum(parts_damage_seg_map[:, :ref_point["x_ref"][0]]) / damaged_area)
            ref_point["is_damaged"].append(np.sum(parts_damage_seg_map[:, ref_point["x_ref"][0]:]) / damaged_area)

    # Post process
    if len(ref_point["is_damaged"]) > 0 :
        ref_point["is_damaged"] = np.round(np.array(ref_point["is_damaged"]), 3).tolist()

    ref_point["ref"] = [ref_point["x_ref"], ref_point["y_ref"]]

    return ref_point

def damage_parts_mapper(data, img_path_name=None, json_path_name=None) :
    if (data["part_res_out"]["check"] and data["damage_res_out"]["check"]) == True :

        ''' 
        damage
            "report":{
                "image": imgdata,
                "damage_state": True,
                "damage_result" :{
                    "damage_name":classes,
                    "scores":scores,
                    "polygons":polygons,
                }
            }
        part
            "report" : {'check': True, 
                        'report': {
                            'front_bumper': {
                                'ids': 2, 
                                'score': 0.9981, 
                                'mask': array([[False, False, False, ..., False, False, False],...,[False, False, False, ..., False, False, False]]), 
                                'mask_expanded': array([[False, False, False, ..., False, False, False],...,[False, False, False, ..., False, False, False]]), 
                                'resized_mask': array([[False, False, False, ..., False, False, False],...,[False, False, False, ..., False, False, False]]), 
                                'inbox_ratio': 0.2522, 
                                'rule_minsize': False, 
                                'rule_focus': False, 
                                'rule_close': False, 
                                'rule_dup': False, 
                                'rule_cont': True, 
                                'rule_out': True, 
                                'rule_adjust': False, 
                                'adjustable': True,
                                "polygon_point" : array([[649 317][654 170][731 172][726 320]])
                                },
					        'front_fender_left':....}
                        }
        '''
        #################################################################
        ## 데이터 변수 선언
        part_res_out = data["part_res_out"]["report"]
        damage_image = data["damage_res_out"]["report"]["image"]
        damage_state = data["damage_res_out"]["report"]["damage_state"]
        #################################################################

        #################################################################
        ## 데미지 여부 확인
        modfiy_damage_polygon_list=[]
        damage_center_ploygon_list=[]
        damage_name = data["damage_res_out"]["report"]["damage_result"]["damage_name"] if (damage_state == True) else []
        damage_scores = data["damage_res_out"]["report"]["damage_result"]["scores"] if (damage_state == True) else []
        damage_polygon = data["damage_res_out"]["report"]["damage_result"]["polygons"] if (damage_state == True) else []

        if damage_state == True:
            # image shape에 damage 좌표 맞추고, center 좌표 추출 / part는 data_res_out에 의해 resize mark가 되어 있고, damage는 안되어 있기 때문에 resize
            for damage_point in damage_polygon:
                modfiy_damage_polygon = change_polygon(damage_image.shape[:2], data["data_res_out"]["image"].shape[:2], damage_point)
                modfiy_damage_polygon_list.append(change_polygon(damage_image.shape[:2], data["data_res_out"]["image"].shape[:2], damage_point))
                damage_center_ploygon = polygon_center(modfiy_damage_polygon)
                damage_center_ploygon_list.append(damage_center_ploygon)
                
            data["damage_res_out"]["report"]["damage_result"]["polygons"] = modfiy_damage_polygon_list
        thresh = model_option["damage"]["box_thres"]
        #################################################################

        #################################################################
        ## part영역에 damage section 분석
        class_list=[]
        score_list=[]
        part_polygon_list=[]

        # mask_for_tool = np.ones(data['data_res_out']['shape'], dtype=np.uint8)

        damage_DATA_CLASSES = ["scratch", "dent", "complex_damage", "broken"]

        for part_name in part_res_out.keys() :
            _resized_mask = part_res_out[part_name]["resized_mask"]
            _part_area = int(np.sum(_resized_mask))
            part_res_out[part_name]["damage"] = dict()

            damage_name_correct_list =[]

            if part_res_out[part_name]["score"] >= thresh:

                modfiy_damage_polygon_correct_list =[]

                class_list.append(part_name)
                score_list.append(part_res_out[part_name]["score"])
                part_polygon_point=make_polygon(part_res_out[part_name]["resized_mask"])
                part_polygon_list.append(part_polygon_point)     
                part_polyin = Polygon(seg_polygon(_resized_mask)) # part의 seg map에 polygon 좌표 생성

                # part 안애 내부 데미지 식별
                for i in range(len(damage_name)):
                    """
                        demage polygon : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 
                        part seg map : binary mask map, 
                        shape을 맞춘 damage center point로 part seg map안에 존재여부 확인 및 각 part 별 각 damage 비율 측정

                    """

                    # part 영역에 damage point 위치 여부 계산 
                    point_desct = Point(damage_center_ploygon_list[i])
                    output_correct = point_desct.within(part_polyin) # part영역에 damage center point가 위치 여부 true/false

                    if output_correct ==True:
                        damage_name_correct_list.append(damage_name[i])
                        part_res_out[part_name]["damage_type"] = damage_name_correct_list

                        modfiy_damage_polygon_correct_list.append(modfiy_damage_polygon_list[i])

                        part_inter_ratio, damage_area = polygon_IOU_v2(part_polyin, modfiy_damage_polygon_list[i]) #겹치는 영역 계산

                        part_res_out[part_name]["damage"][damage_name[i]] = {"area"   : round(damage_area,4),
                                                                            "ratio"  : round(part_inter_ratio,4),
                                                                            "seg_map": modfiy_damage_polygon_list[i]
                                                                            }
                        if damage_name[i] == "complex_damage" :
                            part_res_out[part_name]["damage"][damage_name[i]]["section"] = \
                            get_damaged_section(data, part_name)

                # FOR SUMMARY 전체 데미지, part 별 -> 위 방법은 part 부분에서 걸린 전체 데미지의 영역을 계산하기 때문에 
                """
                    demage polygon : [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], 
                    part seg map : binary mask map, 
                    shape을 맞춘 damage center point로 part seg map안에 존재여부 확인 및 각 part별 총 damage 비율 측정

                """
                correct_poly_list=[]
                for merge_damage_polygon in modfiy_damage_polygon_correct_list:
                    correct_poly_list.append(Polygon(merge_damage_polygon))
                correct_merge_damage_polygons = unary_union(correct_poly_list)
                summary_inter_ratio, full_damage_area = polygon_IOU_v2(part_polyin, correct_merge_damage_polygons)
                if summary_inter_ratio != 0:
                    part_res_out[part_name]["damage"]["summary"] = {"area"   : round(full_damage_area,4),
                                                                    "ratio"  : round(summary_inter_ratio,4),
                                                                    "seg_map": correct_merge_damage_polygons
                                                                    }

            if len(part_res_out[part_name]["damage"].keys()) == 0 :
                del part_res_out[part_name]["damage"]           
                                                                                            

        #################################################################

        #################################################################
        ## image draw
        """
            part image : polygon + score + part name 
            damage image : polygon + score + part name 
            image merage 

        """
        if img_path_name:
            # image save process 
            path_list = ['PART_IMAGE_PATH', 'DAMAGE_IMAGE_PATH', 'PART_DAMAGE_IMAGE_PATH']
            save_dir = []
            for path in path_list:
                result_output_dir='{}'.format(config[path])
                img_dir = os.path.join(result_output_dir, "imgs")
                os.makedirs(img_dir, exist_ok=True)
                save_path = os.path.join(img_dir, img_path_name)
                save_dir.append(save_path)

            # part image
            result_image = draw_result(
                PARTS_DF["part_name"],
                data["data_res_out"]['image'],
                part_polygon_list,
                class_list,
                score_list,
                out_file=None,
                color_list=None,
            )
            cv2.imwrite(save_dir[0],result_image)
            
            # damge image
            result_damage_img = draw_result(
                damage_DATA_CLASSES,
                data["data_res_out"]['image'],
                modfiy_damage_polygon_list,
                damage_name,
                damage_scores,
                out_file=None,
                color_list=None,
            )
            cv2.imwrite(save_dir[1],result_damage_img)
            
            # part_damage merge image
            alpa=0.7
            add_img = cv2.addWeighted(result_image, alpa, result_damage_img, (1-alpa), 0) 
            cv2.imwrite(save_dir[2], add_img)
        #################################################################

        #################################################################
        # json save
        if json_path_name:
            json_path_list = ['JSON_PATH']
            save_json_dir = []
            for path in json_path_list:
                result_output_dir='{}'.format(config[path])
                json_dir = os.path.join(result_output_dir, "json")
                os.makedirs(json_dir, exist_ok=True)
                save_path = os.path.join(json_dir, json_path_name+".json")
                save_json_dir.append(save_path)
            
            save_json(data ,save_json_dir[0])

        #################################################################
        # output json data
        """
            
        """
                            
    return data

# ############################################
# ## image_matching 추가
# ############################################
# def close_damage_parts_mapper(data) :
#     if (data["part_res_out"]["check"] and data["damage_res_out"]["check"]) == True :

#         part_res_out = data["part_res_out"]["report"]
#         damage_seg_map = data["damage_res_out"]["seg_map"]["summary"][-1]
#         thresh = model_option["damage"]["box_thres"]

#         for part_name in part_res_out.keys() :
#             _resized_mask = part_res_out[part_name]["resized_mask"]
#             _part_area = int(np.sum(_resized_mask))
#             part_res_out[part_name]["damage"] = dict()
#             empty_mask = np.zeros(_resized_mask.shape)
            
#             inter_damage_seg_map = copy.deepcopy(damage_seg_map)
#             dam_cls, dam_area = np.unique(inter_damage_seg_map, return_counts=True)
#             for _dam_idx, _dam_cls in enumerate(dam_cls) :
#                 if _dam_cls == 0 : # PASS BG
#                     continue

#                 _dam_cls_name = DAMAGE_DF.loc[DAMAGE_DF["class"] == _dam_cls, "class_name"].values[0]
                    
#                 part_res_out[part_name]["damage"][_dam_cls_name] = {"area"   : int(0),
#                                                                     "ratio"  : round(0.0, 3),
#                                                                     "seg_map": empty_mask}
                
#                 if _dam_cls_name == "complex_damage" :
#                         part_res_out[part_name]["damage"][_dam_cls_name]["section"] = \
#                         get_damaged_section(data, part_name)
                        
#             part_res_out[part_name]["damage"]["summary"] = {"area"   : int(0),
#                                                             "ratio"  : round(0.0, 3),
#                                                             "seg_map": empty_mask}
        
#             if len(part_res_out[part_name]["damage"].keys()) == 0 :
#                 del part_res_out[part_name]["damage"]
                            
#     return data	

def summarize(data) :
    ## data info
    filter_class = check_valid(data["filter_res_out"], "class")
    filter_class_name = FILTER_DF.loc[FILTER_DF["class"] == filter_class].index[0] if filter_class is not None else None

    data_dict = {"url"  : [check_valid(data["data_res_out"], "url")],
                 "image": [check_valid(data["data_res_out"], "image")],
                 "shape": [check_valid(data["data_res_out"], "shape")],
                 "service_code" : [["AISC"+str(x) for x in data["code"]]],
                 "mobile_class" : [check_valid(data["mobile_info"], "class")],
                 "mobile_mode" : [check_valid(data["mobile_info"], "mode")],
                 "mobile_tiny" : [check_valid(data["mobile_info"], "tiny")],
                 "mobile_part_code" : [check_valid(data["mobile_info"], "part_code")],
                 "mobile_part_name" : [check_valid(data["mobile_info"], "part_name")],
                 "aos_part_code": [check_valid(data["aos_info"], "part_code")],
                 "aos_part_name": [check_valid(data["aos_info"], "part_name")],
                 "aos_part_group": [check_valid(data["aos_info"], "part_group")],
                 "filter_class": [filter_class],
                 "filter_class_name": [filter_class_name],
                 "filter_score": [check_valid(data["filter_res_out"], "score")],
                 "filter_probs": [check_valid(data["filter_res_out"], "probs")],
                 "filter_code" : [check_valid(data["filter_res_out"], "code")],
                }

    ## summary info
    summary_dict = None
    if "damage_res_out" in data and data['filter_res_out']['check'] and data['part_res_out']['check']==True:
        part_res_out = data["part_res_out"]["report"]
        if data["damage_res_out"]["check"] == True :
            summary_dict = {"url" : [],
                        "part_name" : [],
                        "part_score": [],
                        "part_inbox_ratio" : [],
                        "part_mask" : [],
                        "part_rule_minsize": [],
                        "part_rule_focus" : [],
                        "part_rule_close" : [],
                        "part_rule_dup": [],
                        "part_rule_cont": [],
                        "part_rule_out": [],
                        "part_rule_adjust": [],
                        "part_adjustable": [],
                        "part_group_replace" : [],
                        "part_damage_type" : [],
                        # "part_code" : [],
                        }
            
            damage_class_list = DAMAGE_DF.loc[DAMAGE_DF["for_dnn"] == True].index.to_list() + ["summary", "result"]
            damage_var_list = np.hstack([["-".join(["damage", x, y]) for y in ["area", "ratio", "mask"]] for x in damage_class_list])
            # print(damage_var_list)
            damage_dict = dict(zip(damage_var_list, [[] for x in damage_var_list]))
            damage_dict.update({"damage-result-section" : [],
                                "damage-result-list" : [],
                                "damage-result-code" : [],
                                "damage-result-ref" : []
                            })
            summary_dict.update(damage_dict)

            # summary res out
            for part_name in part_res_out.keys() :

                summary_dict["url"].append(check_valid(data["data_res_out"], "url"))

                # Parts res out
                summary_dict["part_name"].append(part_name)
                summary_dict["part_score"].append(check_valid(data["part_res_out"]["report"][part_name], "score"))
                summary_dict["part_inbox_ratio"].append(check_valid(data["part_res_out"]["report"][part_name], "inbox_ratio"))
                summary_dict["part_mask"].append(check_valid(data["part_res_out"]["report"][part_name], "resized_mask"))
                summary_dict["part_rule_focus"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_focus"))
                summary_dict["part_rule_close"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_close"))
                summary_dict["part_rule_minsize"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_minsize"))
                summary_dict["part_rule_dup"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_dup"))
                summary_dict["part_rule_cont"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_cont"))
                summary_dict["part_rule_out"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_out"))
                summary_dict["part_rule_adjust"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_adjust"))
                summary_dict["part_adjustable"].append(check_valid(data["part_res_out"]["report"][part_name], "adjustable"))
                summary_dict["part_group_replace"].append(check_valid(data["part_res_out"]["report"][part_name], "group_replace"))
                summary_dict["part_damage_type"].append(check_valid(data["part_res_out"]["report"][part_name], "damage_type"))

                # Damage res out
                dam_part_dict = check_valid(data["part_res_out"]["report"][part_name], "damage")
                dam_part_dict = {} if dam_part_dict is None else dam_part_dict
                no_dam_list = np.setdiff1d(damage_class_list, list(dam_part_dict.keys()))

                for dam_key in [x for x in damage_dict.keys() if "result" not in x] :
                    _, dam_cls, dam_var = dam_key.split("-")

                    dam_var = "seg_map" if dam_var == "mask" else dam_var

                    if (dam_cls not in no_dam_list) & (dam_var not in ["section", "ref"]) :
                        summary_dict[dam_key].append(dam_part_dict[dam_cls][dam_var])

                    else :
                        summary_dict[dam_key].append(None)

                ## make dam result
                for dam_key in [x for x in damage_dict.keys() if "result" in x] :
                    _, dam_cls, dam_var = dam_key.split("-")

                    dam_var = "seg_map" if dam_var == "mask" else dam_var

                    if dam_var not in ["section", "list", "code", "ref", "ratio"] :
                        summary_dict[dam_key] = copy.deepcopy(summary_dict[dam_key.replace("result", "complex_damage")])

                    elif dam_var == "ratio" :
                        summary_dict[dam_key] = copy.deepcopy(summary_dict[dam_key.replace("result", "summary")])

                    elif "complex_damage" in dam_part_dict.keys() :

                        if dam_var == "section" :
                            section_dict = dam_part_dict["complex_damage"]["section"]

                            if section_dict is not None :
                                damage_section = np.array(section_dict["section_name"])[np.array(section_dict["is_damaged"]) > 0].tolist()

                                if len(damage_section) == 0 :
                                    damage_section = None

                            else :
                                damage_section = None

                            summary_dict[dam_key].append(damage_section)

                        elif dam_var == "list" :
                            summary_dict[dam_key].append("complex_damage")

                        elif dam_var == "code" :
                            summary_dict[dam_key].append("DGCL06")

                        elif dam_var == "ref" :
                            damage_ref = check_valid(section_dict, "ref")
                            summary_dict[dam_key].append(damage_ref)

                    else :
                        summary_dict[dam_key].append(None)

    else :
        summary_dict = None

    data["summary"] = {"data_dict" : data_dict,
                       "summary_dict" : summary_dict
                      }
    # print("summary data --", data)
    
    return data

# For MASK RCNN
def mask_iou_checker(part_res_out):
    res_out = False

    if len(part_res_out['masks']) > 0:
        mask_sum = np.nansum(part_res_out['masks'], axis=-1)
        if np.nansum(mask_sum > 0) != 0:
            mask_iou = np.nansum(mask_sum > 1) / np.nansum(mask_sum > 0)
            res_out = mask_iou > 0.5

    return res_out

def count_mask_contour(mask):
    res_out = 0
    if len(np.unique(mask)) == 2:
        mask2image = np.concatenate([np.expand_dims(mask*255, -1).astype(np.uint8)]*3, axis=-1)
        cnts, h = get_contour(image=mask2image, rgb=np.array([255]*3))

        for idx in range(len(cnts)):
            new_data, area = split_contour(data=mask2image, contours=cnts, hierarchy=h, idx=idx)
            if area != 0 :
                res_out = res_out+1

    return res_out

def mask_expand(mask, ksize=None) :
    if ksize is None :
        ksize = int(np.sum(mask)//5000 + 10)
    mask = mask * 255.
    kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (ksize, ksize))

    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = mask == 255.

    return mask

def mask_reshape(mask, shape) :
    h, w, _ = shape
    mask_h, mask_w = mask.shape
    
    if w >= h :
        ratio = h/w
        adj_len = int((mask_h - mask_h * ratio) // 2)
        
        reshaped = mask[adj_len:(mask_h-adj_len), :]
        
    else :
        ratio = w/h
        adj_len = int((mask_w - mask_w * ratio) // 2)
        
        reshaped = mask[:, adj_len:(mask_w-adj_len)]
        
    resized = cv2.resize((reshaped*255).astype(np.uint8), (w, h))
    _, resized = cv2.threshold(resized, 255 / 2, 255, cv2.THRESH_BINARY)
    
    return resized == 255

def get_center(img, y_len=240, x_len=320) :
    H, W = img.shape[:2]
    
    y_min, y_max = int(np.nanmax((H/2-y_len//2, 0))), int(np.nanmin((H/2+(y_len - y_len//2), H))+1)
    x_min, x_max = int(np.nanmax((W/2-x_len//2, 0))), int(np.nanmin((W/2+(x_len - x_len//2), W))+1)
    
    return y_min, y_max, x_min, x_max

def rule_focus(img) :
    # Get center box
    H, W = img.shape[:2]
    y_min, y_max, x_min, x_max = get_center(img, y_len=int(H * 0.4), x_len=int(W * 0.4))

    # Make mask bbox
    mask = copy.deepcopy(img)

    h_indicies = np.where(np.any(mask, axis=0))[0]
    v_indicies = np.where(np.any(mask, axis=1))[0]

    #2021-04-16 모든 값이 0 인지 확인 하는 로직 추가
    #print('h_indicies = {} and v_indicies ={}'.format(h_indicies,v_indicies))
    if(len(h_indicies) > 0 and len(v_indicies) > 0 ) :
        mask[np.nanmin(v_indicies):np.nanmax(v_indicies) + 1, np.nanmin(h_indicies):np.nanmax(h_indicies) + 1] = True
    else :
        return True, 0.0
    # Check in the box
    inbox = mask[y_min:y_max, x_min:x_max]
    #print('inbox = {}'.format(inbox))

    return np.nanmean(inbox) <= 0, np.round(np.nanmean(inbox), 4)

def rule_close(mask):
    in_frame_counts = is_in_frame(mask)

    return in_frame_counts <= 1

def rule_dup(mask, dup_mask):
    merged_mask = mask.astype(np.int) + dup_mask.astype(np.int)

    if np.nansum(merged_mask > 0) != 0:
        mask_iou = np.nansum(merged_mask > 1) / np.nansum(merged_mask > 0)
    else:
        mask_iou = 0

    return mask_iou >= 0.5

def rule_cont(mask) :
    cnt_cont = count_mask_contour(mask=mask)

    return cnt_cont >= 4

def make_sheet_l(x):
    out = False
    if x is not None:
        if len(x) == 5:
            out = (np.argsort(x)[-2] == 3)

    return out

def merge_output(data, aos_list=[]) :
    # Output Control Params
    filter_mobile=True
    filter_aos=True

    # Summary output
    _data_dict = []
    _summary_dict = []
    for inner_data in data :
        if inner_data["summary"]["data_dict"] is not None :
            _data_dict.append(pd.DataFrame(inner_data["summary"]["data_dict"]))
           
        if inner_data["summary"]["summary_dict"] is not None :
            _summary_dict.append(pd.DataFrame(inner_data["summary"]["summary_dict"]))

    # Build data frame
    if len(_data_dict) > 0 :
        data_df = pd.concat(_data_dict, axis=0).sort_values("url").reset_index(drop=True)
    else :
        data_df = pd.DataFrame(_data_dict)

    if len(_summary_dict) > 0 :
        summary_df = pd.concat(_summary_dict, axis=0).sort_values("url").reset_index(drop=True)
    else :
        summary_df = pd.DataFrame(_summary_dict)

    # Make result
    ## Make variables
    if len(data_df) > 0 :
        data_df["file_index"] = range(data_df.shape[0])
 
    ## Check summary_df  # 손상 유형 결과 확인
    if len(summary_df) > 0 :
        ## FILL NA
        summary_df.loc[:, "damage-result-section"] = \
        summary_df["damage-result-section"].apply(lambda x : ["DGRGXX"] if x is None else x)
        summary_df.loc[pd.isna(summary_df["damage-result-list"]), "damage-result-list"] = "no_complex"
        summary_df.loc[pd.isna(summary_df["damage-result-code"]), "damage-result-code"] = "DGCLXX"

        ## Make variables
        summary_df["index"] = range(summary_df.shape[0])
        summary_df = summary_df.merge(data_df.loc[:, ["url", "file_index"]], how="left", on="url")

        ## part Rule handling
        summary_df["merging_selected"] = True
        summary_df.loc[(summary_df["merging_selected"] == True) & \
                      (summary_df["part_rule_out"] == True), "merging_selected"] = False


        summary_df["damage-summary-ratio"].fillna(0, inplace=True)

       
        ## Mobile Filtering
        _mobile_df = data_df.loc[(pd.notna(data_df["mobile_part_code"]))&
                                 (data_df["mobile_part_code"]!="DGPT99"), ["url", "mobile_part_code"]]


        # filter_mobile 플래그 변경 _20210401
        if (filter_aos) & (len(aos_list) > 0):
            filter_mobile = False

        if (len(_mobile_df) > 0) & (filter_mobile):
            summary_df = summary_df.merge(_mobile_df, how='left')
            summary_df.loc[(summary_df["merging_selected"] == True) & (pd.notna(summary_df["mobile_part_code"])) & \
                          (summary_df["part_code"] != summary_df["mobile_part_code"]), "merging_selected"] = False

            ##로직추가 20210325 LKS (mobile part code와 다르면 damage-summary-ratio값 0으로 변경)
            summary_df.loc[((summary_df["mobile_part_code"] != None) & (summary_df["part_code"] != summary_df["mobile_part_code"] )),"damage-summary-ratio"] = 0

            summary_df = summary_df.drop(columns=["mobile_part_code"])
        
        
        ### Ranking 매기는 부분
        summary_df["merging_rank"] = summary_df.groupby("part_name")["damage-summary-ratio"].rank("dense", ascending=False).astype(int)
        
        check_dup = ~summary_df.duplicated(["part_name", "damage-summary-ratio"], keep='first')

        summary_df["summary_result"] = (summary_df["merging_rank"] == 1) & (summary_df["merging_selected"] == True) & (check_dup)

        # ## AOS Filtering
        # if (filter_aos) & (len(aos_list) > 0):
        #         delete_idx = summary_df.loc[:, "part_code"].apply(lambda x: x not in aos_list)
        #         summary_df.loc[delete_idx, "summary_result"] = False


        ## Make Sheeting_L
        summary_df["summary_adjust"] = False # Removed rule

        ## Make missing mobile_part_code
        mobile_url_list=data_df.loc[(data_df["mobile_class"] == "parts") & \
                                    (pd.isna(data_df["mobile_part_code"])), "url"].values
        for _url in mobile_url_list:
            mobile_part_name_list = summary_df.loc[summary_df["url"] == _url].sort_values(["damage-result-ratio", "merging_selected", "merging_rank"], ascending=False)["part_name"].values
            if len(mobile_part_name_list) > 0:
                part_name = mobile_part_name_list[0]
                part_code = PARTS_DF.loc[PARTS_DF["part_name"] == part_name, "part_code"].values.item()

                data_df.loc[(data_df["mobile_class"] == "parts") & (pd.isna(data_df["mobile_part_code"])), \
                            "mobile_part_name"] = part_name
                data_df.loc[(data_df["mobile_class"] == "parts") & (pd.isna(data_df["mobile_part_code"])), \
                            "mobile_part_code"] = part_code

    # Make output dict
    ## Make ai_result
    if len(summary_df) > 0 :
        _ai_result_df = summary_df.loc[summary_df["summary_result"] == True, \
                                      ["index", "file_index", "url", \
                                       "part_name", "part_score", "part_damage_type", \
                                       "part_mask", "damage-summary-mask", \
                                       "damage-result-section", "damage-summary-ratio",\
                                       "damage-result-list", "damage-result-code"]]
        _ai_result_df = _ai_result_df.sort_values("damage-summary-ratio", ascending=False)
        _ai_result_df = _ai_result_df.round(4)
        _ai_result_df = _ai_result_df.fillna("")
        _ai_result_df.index = ["part_" + str(x) for x in range(len(_ai_result_df.index))]

        _ai_result_dict = _ai_result_df.transpose().to_dict()

        # change Polygon object to list
        for key, value in _ai_result_dict.items():

            poly_obj = _ai_result_dict[key]["damage-summary-mask"]

            if poly_obj == "":
                _ai_result_dict[key]["damage-summary-mask"] = list([])
            else:
                _ai_result_dict[key]["damage-summary-mask"] = list(poly_obj.exterior.coords)[:4]

            mask_obj = _ai_result_dict[key]["part_mask"]
            if mask_obj == "":
                _ai_result_dict[key]["part_mask"] = list([])
            else:
                size, rle = encode_mask_to_rle(_ai_result_dict[key]["part_mask"])

                _ai_result_dict[key]["part_mask"] = {"size":size,"counts":rle.decode("utf-8")}


    else :
        _ai_result_dict = {}

    ## Make ai_info
    _ai_info_dict = dict()
    for ai_info_idx, url_idx in enumerate(data_df["url"]) :
        _key = "file_" + str(ai_info_idx)

        _file_df = data_df.loc[data_df["url"] == url_idx, ["file_index", "url", "shape", "service_code", \
                                                           "mobile_class", "mobile_mode", "mobile_tiny", \
                                                           "mobile_part_code", "mobile_part_name", \
                                                           "aos_part_code", "aos_part_name", "aos_part_group", \
                                                            ]]
        _file_df = _file_df.round(4)
        _file_df = _file_df.fillna("")
        _file_df.index = ["file"]

        _filter_df = data_df.loc[data_df["url"] == url_idx, ["filter_class", "filter_class_name", \
                                                             "filter_score", "filter_probs", "filter_code"]]
        _filter_df = _filter_df.round(4)
        _filter_df = _filter_df.fillna("")
        _filter_df.index = ["filter"]

        if len(summary_df) > 0 :
            _summary_df = summary_df.loc[summary_df["url"] == url_idx, \
                                       [x for x in summary_df.columns if all([y not in x for y in ["image", "mask", "url", "ref", "file_index"]])]]
            _summary_df = _summary_df.round(4)
            _summary_df = _summary_df.fillna("")
            _summary_df.index = ["part_" + str(x) for x in range(len(_summary_df.index))]
        else :
            _summary_df = pd.DataFrame([])

        _ai_info_dict[_key] = dict()
        _ai_info_dict[_key].update({"file" : list(_file_df.transpose().to_dict().values())})
        _ai_info_dict[_key].update({"filter" : list(_filter_df.transpose().to_dict().values())})
        _ai_info_dict[_key].update({"part" : list(_summary_df.transpose().to_dict().values())})
        #_file_df.to_csv('106_{}_file_df.csv'.format(ai_info_idx),mode='w')
    ## Make etc
    #print('data_num= {}'.format(len(data_df)))
    _misc_dict = {"data_count" : {"input" : len(data_df),
                                  "filter" : len(data_df.loc[(pd.notna(data_df["filter_score"])) & \
                                                             (data_df["filter_class"] != 0)]),
                                  "error" : len(data_df.loc[pd.isnull(data_df["image"])])
                                  }
                  }

    ## Make output_dict
    output_dict = {"misc_info" : _misc_dict,
                   "ai_result" : list(_ai_result_dict.values()),
                   "ai_info"   : list(_ai_info_dict.values())
                  }
    
    return output_dict, data_df

def gen_valid_image(data_df, repair_df) :
    
    ## 손상심도화면에서 보여주기 위해 파트와 손상 정보를 이미지에 표시하여 버퍼에 저장

    if repair_df.shape[0] > 0 :
        report_df = repair_df.sort_values(by=["url", "merging_score"], ascending=False).reset_index(drop=True)
        report_df = report_df.loc[(report_df["repair_result"] == True) & (report_df["repair_class"] != 0)]
        report_df["valid_image"]=None
        report_df["valid_image_b64"]=None

        image_dict = dict()
        for idx in report_df.index.to_numpy() :
            _url = report_df.loc[report_df.index == idx, "url"].values[0]
            _part_name = report_df.loc[report_df.index == idx, "part_name"].values[0]

            # 파트영역 마스크 적용
            
            _image = np.array(data_df.loc[data_df["url"] == _url, "image"].values[0], dtype=np.uint8)
            _damage_seg_map = np.array(report_df.loc[report_df.index == idx, "damage-summary-mask"].values[0])
            _part_masks = report_df.loc[report_df.index == idx, 'part_mask']
            for idxx in _part_masks.index.to_numpy():
                _temp_part_masks = _part_masks.loc[_part_masks.index==idxx]
                for _temp_part_mask in _temp_part_masks:
                    _temp_part_mask = _temp_part_mask * 1
                    _part = segmap2image(_temp_part_mask, PARTS_DF, df_var="model_cls")
                    _image = apply_mask(_image, _part)

            # 손상영역 마스크 적용
            # 마스크를 0-1로 clip하여 손상종류에 상관없이 한가지 색상으로 적용
            
            if len(_damage_seg_map.shape) == 0 :
                _damage_seg_map = np.zeros(list(_image.shape)[:2], dtype=np.uint8)
            elif len(_damage_seg_map.shape) == 3 :
                _damage_seg_map = _damage_seg_map[-1]
            
            _damage_seg_map = cv2.resize(_damage_seg_map, (_image.shape[1], _image.shape[0]), interpolation=cv2.INTER_NEAREST)
            _damage = segmap2image(np.clip(_damage_seg_map, 0, 1), DAMAGE_DF)
            _valid = apply_mask(_image, _damage)

            _valid_image = Image.fromarray(_valid)
            buffer = io.BytesIO()
            _valid_image.save(buffer, format='jpeg')


            # Insert result
            image_dict['file{}'.format(idx)] = ('file{}'.format(idx), buffer, 'image/jpg')
            report_df.loc[report_df.index == idx, "valid_image"] = 'file{}'.format(idx)


        # rewrite dictionary
        _ai_result_df=report_df.loc[report_df["repair_result"] == True, \
                                              ["index", "file_index", "url", \
                                               "part_name", "part_score", "part_code", \
                                               "repair_class_name", "repair_score", "repair_code", \
                                               "damage-result-ratio", "damage-result-section", \
                                               "damage-result-list", "damage-result-code", "valid_image"]]
                                               # "damage-result-list", "damage-result-code", "valid_image", "valid_image_b64"]]

        _ai_result_df = _ai_result_df.sort_values("repair_code", ascending=False)
        _ai_result_df = _ai_result_df.round(4)
        _ai_result_df = _ai_result_df.fillna("")
        _ai_result_df.index = ["part_" + str(x) for x in range(len(_ai_result_df.index))]
        _ai_result_dict = _ai_result_df.transpose().to_dict()
        output_dict = {"ai_result": list(_ai_result_dict.values()),
                       "image_dict": image_dict}

    else :
        output_dict = None

    return output_dict

# For deeplab
def seg_map_reshape(data, shape) :
    origin_dtype = data.dtype
    h, w, _ = shape
    
    new_data = np.zeros((h, w), dtype=np.int32)
    
    for model_cls in np.unique(data) :
        cls_map = cv2.resize(((data == model_cls)*255).astype(np.uint8), (w, h))
        _, resized = cv2.threshold(cls_map, 255 / 2, 255, cv2.THRESH_BINARY)
        
        new_data[np.where(resized == 255)] = model_cls

    return new_data.astype(origin_dtype)

def get_contour(image, rgb) :
    value = np.array(rgb)
    mask_bi = cv2.inRange(image, value-1, np.minimum(value+1, np.array([255, 255 ,255])))

    kernel = np.ones((2, 2), np.uint8)
    mask_bi = cv2.morphologyEx(mask_bi, cv2.MORPH_OPEN, kernel)

    try :
        cnts, h = cv2.findContours(mask_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    except :
        _, cnts, h = cv2.findContours(mask_bi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
    return cnts, h

def split_contour(data, contours, hierarchy, idx) :
    new_data = np.zeros(data.shape[:2], dtype=np.uint8)

    if hierarchy[0][idx][-1] == -1 :    
        new_data = cv2.drawContours(new_data, contours, idx, 1, -1)
    
    new_data = new_data.astype(np.bool)
    area = np.sum(new_data)
    
    return new_data, area
    
# For tensorpack
def np_area(boxes):
    """
    Computes area of boxes.

    Args:
        boxes: Numpy array with shape [N, 4] holding N boxes

    Returns:
        a numpy array with shape [N*1] representing box areas
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def draw_text(img, pos, text, color, font_scale=0.4):
    """
    Draw text on an image.

    Args:
        pos (tuple): x, y; the position of the text
        text (str):
        font_scale (float):
        color (tuple): a 3-tuple BGR color in [0, 255]
    """
    img = img.astype(np.uint8)
    x0, y0 = int(pos[0]), int(pos[1])
    # Compute text size.
    font = cv2.FONT_HERSHEY_COMPLEX
    ((text_w, text_h), _) = cv2.getTextSize(text, font, font_scale, 1)
    # Place text background.
    if x0 + text_w > img.shape[1]:
        x0 = img.shape[1] - text_w
    if y0 - int(1.15 * text_h) < 0:
        y0 = int(1.15 * text_h)
    back_topleft = x0, y0 - int(1.3 * text_h)
    back_bottomright = x0 + text_w, y0
    cv2.rectangle(img, back_topleft, back_bottomright, color, -1)
    # Show text.
    text_bottomleft = x0, y0 - int(0.25 * text_h)
    cv2.putText(img, text, text_bottomleft, font, font_scale, (50, 50, 50), lineType=cv2.LINE_AA)

    return img

#####################
# get_part_damage_mask_data
# 부품 마스크 데이터 및 손상 마스크 데이터 추출 함수
#####################
def get_part_damage_mask_data(report, is_matching=False):
    part_res_out_report = None
    if is_matching:
        part_res_out_report = report["part_res_out"]['report']
    else:
        part_res_out_report = report["org_part_res_out"]['report']
            
    #원거리 이미지 부품 마스크 가져오기
    part_mask_data_list = []
    for _, part_name in enumerate(part_res_out_report):
        part_score = part_res_out_report[part_name]['score']
        
        '''Debug'''
        '''
        if is_matching:
            print("IM-O 근접 부품 : ", part_name, ", 부품 스코어 : ", round(part_score,3))
        else:
            print("IM-X 근접 부품 : ", part_name, ", 부품 스코어 : ", round(part_score,3))
        '''
        #근접 여부 조건과 동일하게    
        if part_score >= SDS_CONFIG_CLOSE_PART_SCORE:
            resized_mask = part_res_out_report[part_name]['resized_mask']
            mask = np.asanyarray(resized_mask, dtype= np.uint8)
            #400, 300 기준
            r_mask = cv2.resize(mask, (int(SDS_CONFIG_IM_WIDTH/SDS_IM_DIVIDE_RATIO),int(SDS_CONFIG_IM_HEIGHT/SDS_IM_DIVIDE_RATIO)), interpolation=cv2.INTER_NEAREST)
            part_mask_data = dict()
            part_mask_data['part_name'] = part_name
            part_mask_data['mask_image'] = r_mask
            
            #Debug
            part_mask_data['part_score'] = part_score
            part_mask_data_list.append(part_mask_data)
    
    #근접 손상 마스크 가져오기
    damage_mask_data_list = []
    for _, damage_name in enumerate(report["damage_res_out"]["seg_map"].keys()):
        if damage_name == 'summary' : continue
        damage_mask = report["damage_res_out"]["seg_map"][damage_name]
        
        damage_mask_data = dict()
        damage_mask_data['damage_name'] = damage_name
     
        idx = damage_mask.shape[0]-1
        damage_mask[idx][damage_mask[idx] != 0] = 1
        damage_mask2 = np.asarray(damage_mask[idx], dtype=np.uint8)
        #400,300 기준
        r_im_damage_mask2 = cv2.resize(damage_mask2, (int(SDS_CONFIG_IM_WIDTH/SDS_IM_DIVIDE_RATIO),int(SDS_CONFIG_IM_HEIGHT/SDS_IM_DIVIDE_RATIO)), interpolation=cv2.INTER_NEAREST)
        damage_mask_data['mask_image'] = r_im_damage_mask2
        
        '''Damage Score'''
        temp_damage_scores = [x["score"] for x in report["damage_res_out"]["report"].values() if x["class_name"] == damage_name]
        if len(temp_damage_scores) == 0:
            damage_mask_data['damage_score'] = 0.0
        else :
            damage_mask_data['damage_score'] = max(temp_damage_scores)
        damage_mask_data_list.append(damage_mask_data)
    
    return part_mask_data_list, damage_mask_data_list


#####################
# far_close_rule
# 이미지별 원거리 - 근거리 분류 함수
#####################
def far_close_rule(report, max_part_area, part_code_list):
    
    #원거리 분류 기본 룰
    #1. 이미지 별 0.8 스코어 이상의 파트 영역 중 하나라도 16% 넘는게 없다면 원거리
    #2. 이미지 별 0.8 스코어 이상의 파트가 6개 이상인 경우 원거리
    far_image_rule_1 = report['filter_res_out']['check'] and True if max_part_area < 0.16 else False 
    
    far_image_rule_1_1 = report['filter_res_out']['check'] and len(part_code_list) >= 6

    far_image_rule = far_image_rule_1 + far_image_rule_1_1

    part_except_code_set = set(part_code_list)
    
    #리어 방향 원거리 분류 기준
    #1. 리어범퍼, 리어램프우, 리어램프좌, 트렁크 혹은 백도어가 나오면 원거리  
    #2. 리어범퍼, 리어램프우, 리어휀더우, 트렁크 혹은 백도어가 나오면 원거리
    #3. 리어범퍼, 리어램프좌, 리어휀더좌, 트렁크 혹은 백도어가 나오면 원거리
    exception_rule_1 = part_except_code_set.issuperset({'rear_bumper', 'rear_lamp_right', 'rear_lamp_left'}) and ('trunk' in part_code_list or 'back_door' in part_code_list)
    
    exception_rule_1_1 = part_except_code_set.issuperset({'rear_bumper', 'rear_lamp_right', 'rear_fender_right'}) and ('trunk' in part_code_list or 'back_door' in part_code_list)

    exception_rule_1_2 = part_except_code_set.issuperset({'rear_bumper', 'rear_lamp_left', 'rear_fender_left'}) and ('trunk' in part_code_list or 'back_door' in part_code_list)

    exception_rule_rear = report['part_res_out']['check'] * (exception_rule_1 + exception_rule_1_1 + exception_rule_1_2)

    #정면 방향 원거리 분류 기준
    #1. 프론트범퍼, 프론트램프우, 프론트램프좌 나오면 원거리  
    #2. 후드, 프론트램프우, 프론트램프좌 나오면 원거리  
    #3. 프론트램프우, 프론트휀더우, 사이드미러우 나오면 원거리  
    #4. 프론트램프좌, 프론트휀더좌, 사이드미러좌 나오면 원거리  
    exception_rule_2 = part_except_code_set.issuperset({'front_bumper', 'front_lamp_right', 'front_lamp_left'})
    exception_rule_2_1 = part_except_code_set.issuperset({'hood', 'front_lamp_right', 'front_lamp_left'})
    exception_rule_2_2 = part_except_code_set.issuperset({'front_lamp_right', 'front_fender_right', 'side_mirror_right'})
    exception_rule_2_3 = part_except_code_set.issuperset({'front_lamp_left', 'front_fender_left', 'side_mirror_left'})

    exception_rule_front = report['part_res_out']['check'] * (exception_rule_2 + exception_rule_2_1 + exception_rule_2_2 + exception_rule_2_3)
    #측면 방향 원거리 분류 기준
    #1. 프론트휀더우, 프론트도어우, 리어도어우 나오면 원거리  
    #2. 프론트도어우, 리어도어우, 리어휀더우 나오면 원거리  
    #3. 프론트범퍼, 프론트휀더우, 프론트도어우 나오면 원거리
    #4. 리어도어우, 리어휀더우, 리어범퍼 나오면 원거리 
    exception_rule_3 = part_except_code_set.issuperset({'front_fender_right', 'front_door_right', 'rear_door_right'})
    exception_rule_3_1 = part_except_code_set.issuperset({'front_door_right', 'rear_door_right', 'rear_fender_right'})
    exception_rule_3_2 = part_except_code_set.issuperset({'front_bumper', 'front_door_right', 'front_fender_right'})
    exception_rule_3_3 = part_except_code_set.issuperset({'rear_bumper', 'rear_door_right', 'rear_fender_right'})
    
    exception_rule_rightside = report['part_res_out']['check'] * (exception_rule_3 + exception_rule_3_1+exception_rule_3_2+exception_rule_3_3)

    #1. 프론트휀더좌, 프론트도어좌, 리어도어좌 나오면 원거리  
    #2. 프론트도어좌, 리어도어좌, 리어휀더좌 나오면 원거리  
    #3. 프론트범퍼, 프론트휀더좌, 프론트도어좌 나오면 원거리
    #4. 리어도어좌, 리어휀더좌, 리어범퍼 나오면 원거리 
    exception_rule_4 = part_except_code_set.issuperset({'front_fender_left', 'front_door_left', 'rear_door_left'})
    exception_rule_4_1 = part_except_code_set.issuperset({'front_door_left', 'rear_door_left', 'rear_fender_left'})
    exception_rule_4_2 = part_except_code_set.issuperset({'front_bumper', 'front_door_left', 'front_fender_left'})
    exception_rule_4_3 = part_except_code_set.issuperset({'rear_bumper', 'rear_door_left', 'rear_fender_left'})
    
    exception_rule_leftside = report['part_res_out']['check'] * (exception_rule_4 + exception_rule_4_1 + exception_rule_4_2 + exception_rule_4_3)

    far_rule = (far_image_rule + exception_rule_rear + exception_rule_front +exception_rule_rightside + exception_rule_leftside)
      
    return far_rule
