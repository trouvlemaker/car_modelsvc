from app.ai.config import *

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import copy
import json
import io
import requests
json.encoder.FLOAT_REPR = lambda f: ("%.4f" % f)

from decimal import Decimal
from datetime import datetime


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

def damage_parts_mapper(data) :
    if data["part_res_out"]["check"] == data["damage_res_out"]["check"] == True :

        part_res_out = data["part_res_out"]["report"]
        damage_seg_map = data["damage_res_out"]["seg_map"]["summary"][-1]
        thresh = model_option["damage"]["box_thres"]

        for part_name in part_res_out.keys() :
            _resized_mask = part_res_out[part_name]["resized_mask"]
            _part_area = int(np.sum(_resized_mask))
            part_res_out[part_name]["damage"] = dict()

            if np.sum((damage_seg_map > 0) & _resized_mask) > 0 :
                inter_damage_seg_map = copy.deepcopy(damage_seg_map)
                inter_damage_seg_map[np.where(_resized_mask == False)] = 0

                dam_cls, dam_area = np.unique(inter_damage_seg_map, return_counts=True)
                dam_ratio = dam_area / _part_area

                for _dam_idx, _dam_cls in enumerate(dam_cls) :
                    if _dam_cls == 0 : # PASS BG
                        continue

                    _dam_cls_name = DAMAGE_DF.loc[DAMAGE_DF["class"] == _dam_cls, "class_name"].values[0]

                    part_res_out[part_name]["damage"][_dam_cls_name] = {"area"   : int(dam_area[_dam_idx]),
                                                                        "ratio"  : round(dam_ratio[_dam_idx], 3),
                                                                        "seg_map": inter_damage_seg_map}

                    if _dam_cls_name == "complex_damage" :
                        part_res_out[part_name]["damage"][_dam_cls_name]["section"] = \
                        get_damaged_section(data, part_name)

                # FOR SUMMARY
                summary_area = int(np.sum(inter_damage_seg_map > 0))
                summary_ratio = round(np.sum(inter_damage_seg_map > 0) / _part_area, 3)

                part_res_out[part_name]["damage"]["summary"] = {"area"   : summary_area,
                                                                "ratio"  : summary_ratio,
                                                                "seg_map": inter_damage_seg_map}

            if len(part_res_out[part_name]["damage"].keys()) == 0 :
                del part_res_out[part_name]["damage"]
                            
    return data

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

    ## repair info
    if data["repair_res_out"]["check"] == True :
        repair_dict = {"url" : [],
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
                       "part_code" : [],
                       "cnn_class" : [],
                       "cnn_score" : [],
                       "cnn_probs" : [],
                       "cnn_code" : [],
                       "cnn_class_name" : [],
                       "repair_data" : [],
                       "repair_image" : [],
                       "repair_class" : [],
                       "repair_score" : [],
                       "repair_probs" : [],
                       "repair_code" : [],
                       "repair_class_name" : [],
                      }

        damage_class_list = DAMAGE_DF.loc[DAMAGE_DF["for_dnn"] == True].index.to_list() + ["summary", "result"]
        damage_var_list = np.hstack([["-".join(["damage", x, y]) for y in ["area", "ratio", "mask"]] for x in damage_class_list])
        damage_dict = dict(zip(damage_var_list, [[] for x in damage_var_list]))
        damage_dict.update({"damage-result-section" : [],
                            "damage-result-list" : [],
                            "damage-result-code" : [],
                            "damage-result-ref" : []
                           })
        repair_dict.update(damage_dict)

        # CNN res out
        for cnn_data in list(data["cnn_res_out"]["report"].values()) :

            repair_dict["cnn_class"].append(check_valid(cnn_data, "class"))
            repair_dict["cnn_score"].append(check_valid(cnn_data, "score"))
            repair_dict["cnn_probs"].append(check_valid(cnn_data, "probs"))
            repair_dict["cnn_code"].append(check_valid(cnn_data, "repair_code"))
            repair_dict["cnn_class_name"].append(check_valid(cnn_data, "class_name"))

        # Repair res out
        for repair_data in list(data["repair_res_out"]["report"].values()) :

            repair_dict["url"].append(check_valid(data["data_res_out"], "url"))

            # Parts res out
            part_name = check_valid(repair_data, "part_name")

            repair_dict["part_name"].append(part_name)
            repair_dict["part_score"].append(check_valid(data["part_res_out"]["report"][part_name], "score"))
            repair_dict["part_inbox_ratio"].append(check_valid(data["part_res_out"]["report"][part_name], "inbox_ratio"))
            repair_dict["part_mask"].append(check_valid(data["part_res_out"]["report"][part_name], "resized_mask"))
            repair_dict["part_rule_focus"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_focus"))
            repair_dict["part_rule_close"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_close"))
            repair_dict["part_rule_minsize"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_minsize"))
            repair_dict["part_rule_dup"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_dup"))
            repair_dict["part_rule_cont"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_cont"))
            repair_dict["part_rule_out"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_out"))
            repair_dict["part_rule_adjust"].append(check_valid(data["part_res_out"]["report"][part_name], "rule_adjust"))
            repair_dict["part_adjustable"].append(check_valid(data["part_res_out"]["report"][part_name], "adjustable"))
            repair_dict["part_group_replace"].append(check_valid(data["part_res_out"]["report"][part_name], "group_replace"))

            # Damage res out
            dam_part_dict = check_valid(data["part_res_out"]["report"][part_name], "damage")
            dam_part_dict = {} if dam_part_dict is None else dam_part_dict
            no_dam_list = np.setdiff1d(damage_class_list, list(dam_part_dict.keys()))

            for dam_key in [x for x in damage_dict.keys() if "result" not in x] :
                _, dam_cls, dam_var = dam_key.split("-")

                dam_var = "seg_map" if dam_var == "mask" else dam_var

                if (dam_cls not in no_dam_list) & (dam_var not in ["section", "ref"]) :
                    repair_dict[dam_key].append(dam_part_dict[dam_cls][dam_var])

                else :
                    repair_dict[dam_key].append(None)

            ## make dam result
            for dam_key in [x for x in damage_dict.keys() if "result" in x] :
                _, dam_cls, dam_var = dam_key.split("-")

                dam_var = "seg_map" if dam_var == "mask" else dam_var

                if dam_var not in ["section", "list", "code", "ref", "ratio"] :
                    repair_dict[dam_key] = copy.deepcopy(repair_dict[dam_key.replace("result", "complex_damage")])

                elif dam_var == "ratio" :
                    repair_dict[dam_key] = copy.deepcopy(repair_dict[dam_key.replace("result", "summary")])

                elif "complex_damage" in dam_part_dict.keys() :

                    if dam_var == "section" :
                        section_dict = dam_part_dict["complex_damage"]["section"]

                        if section_dict is not None :
                            damage_section = np.array(section_dict["section_name"])[np.array(section_dict["is_damaged"]) > 0].tolist()

                            if len(damage_section) == 0 :
                                damage_section = None

                        else :
                            damage_section = None

                        repair_dict[dam_key].append(damage_section)

                    elif dam_var == "list" :
                        repair_dict[dam_key].append("complex_damage")

                    elif dam_var == "code" :
                        repair_dict[dam_key].append("DGCL06")

                    elif dam_var == "ref" :
                        damage_ref = check_valid(section_dict, "ref")
                        repair_dict[dam_key].append(damage_ref)

                else :
                    repair_dict[dam_key].append(None)

            # Repair res out
            repair_dict["repair_data"].append(check_valid(repair_data, "data"))
            repair_dict["repair_image"].append(check_valid(repair_data, "image"))
            repair_dict["repair_class"].append(check_valid(repair_data, "class"))
            repair_dict["repair_score"].append(check_valid(repair_data, "score"))
            repair_dict["repair_probs"].append(check_valid(repair_data, "probs"))
            repair_dict["part_code"].append(check_valid(repair_data, "part_code"))
            repair_dict["repair_code"].append(check_valid(repair_data, "repair_code"))
            repair_dict["repair_class_name"].append(check_valid(repair_data, "class_name"))

    else :
        repair_dict = None

    data["summary"] = {"data_dict" : data_dict,
                       "repair_dict" : repair_dict
                      }
    
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
        ksize = np.sum(mask)//5000 + 10

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
    
    mask[np.nanmin(v_indicies):np.nanmax(v_indicies) + 1, np.nanmin(h_indicies):np.nanmax(h_indicies) + 1] = True

    # Check in the box
    inbox = mask[y_min:y_max, x_min:x_max]
    
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
    _repair_dict = []

    for inner_data in data :
        if inner_data["summary"]["data_dict"] is not None :
            _data_dict.append(pd.DataFrame(inner_data["summary"]["data_dict"]))

        if inner_data["summary"]["repair_dict"] is not None :
            _repair_dict.append(pd.DataFrame(inner_data["summary"]["repair_dict"]))

    # Build data frame
    if len(_data_dict) > 0 :
        data_df = pd.concat(_data_dict, axis=0).sort_values("url").reset_index(drop=True)
    else :
        data_df = pd.DataFrame(_data_dict)

    if len(_repair_dict) > 0 :
        repair_df = pd.concat(_repair_dict, axis=0).sort_values("url").reset_index(drop=True)
    else :
        repair_df = pd.DataFrame(_repair_dict)

    # Make result
    ## Make variables
    if len(data_df) > 0 :
        data_df["file_index"] = range(data_df.shape[0])

    ## Check repair_df
    if len(repair_df) > 0 :
        ## FILL NA
        ### damage
        repair_df.loc[:, "damage-result-section"] = \
        repair_df["damage-result-section"].apply(lambda x : ["DGRGXX"] if x is None else x)
        repair_df.loc[pd.isna(repair_df["damage-result-list"]), "damage-result-list"] = "no_complex"
        repair_df.loc[pd.isna(repair_df["damage-result-code"]), "damage-result-code"] = "DGCLXX"

        ## Make variables
        repair_df["index"] = range(repair_df.shape[0])
        repair_df = repair_df.merge(data_df.loc[:, ["url", "file_index"]], how="left", on="url")

        ## Remove repair normal class
        repair_df["merging_selected"] = False
        repair_df.loc[(pd.notna(repair_df["repair_code"])) & \
                      (repair_df["repair_code"] != "RPCL00"), "merging_selected"] = True

        ## part Rule handling
        repair_df.loc[(repair_df["merging_selected"] == True) & \
                      (repair_df["part_rule_out"] == True), "merging_selected"] = False

        ## Mobile Filtering
        _mobile_df = data_df.loc[(pd.notna(data_df["mobile_part_code"]))&
                                 (data_df["mobile_part_code"]!="DGPT99"), ["url", "mobile_part_code"]]
        if (len(_mobile_df) > 0) & (filter_mobile):
            repair_df = repair_df.merge(_mobile_df, how='left')
            repair_df.loc[(repair_df["merging_selected"] == True) & (pd.notna(repair_df["mobile_part_code"])) & \
                          (repair_df["part_code"] != repair_df["mobile_part_code"]), "merging_selected"] = False

            repair_df = repair_df.drop(columns=["mobile_part_code"])

        ## Merge out by parts
        repair_df["merging_score"] = 0

        repair_df.loc[repair_df["merging_selected"] == True, "merging_score"] = \
        repair_df.loc[:, ["part_inbox_ratio", "repair_score", "repair_class"]].fillna(0).apply(lambda row : (row["part_inbox_ratio"]+row["repair_score"]) * row["repair_class"], axis=1)

        repair_df["merging_rank"] = repair_df.groupby("part_name")["merging_score"].rank("dense", ascending=False).astype(int)

        check_dup = ~repair_df.duplicated(["part_name", "merging_score"], keep='first')

        repair_df["repair_result"] = (repair_df["merging_rank"] == 1) & (repair_df["merging_selected"] == True) & (check_dup)

        ## AOS Filtering
        if (filter_aos) & (len(aos_list) > 0):
                delete_idx = repair_df.loc[:, "part_code"].apply(lambda x: x not in aos_list)
                repair_df.loc[delete_idx, "repair_result"] = False

        ## Make Sheeting_L
        repair_df["repair_adjust"] = False # Removed rule

        ## Part_adjustable filtering
        result_part_list = np.unique(repair_df.loc[repair_df["repair_result"] == True, "part_code"].values).tolist()
        part_adjust_target = np.setdiff1d(aos_list, result_part_list)

        if len(part_adjust_target) > 0 :
            part_group_list = np.unique(PARTS_DF.loc[PARTS_DF.apply(lambda x: x["part_code"] in \
                                                                    part_adjust_target, axis=1), "part_group_upper"].values)
            part_adjust_list = PARTS_DF.loc[PARTS_DF["part_group_upper"].apply(lambda x: x in part_group_list), "part_code"].values

            repair_df.loc[(repair_df["part_adjustable"] == True) & \
                              (repair_df["part_code"].apply(lambda x: x not in part_adjust_list)), \
                          "part_adjustable"] = False
        else:
            repair_df.loc[repair_df["part_adjustable"] == True, "part_adjustable"] = False

        ## Make missing mobile_part_code
        mobile_url_list=data_df.loc[(data_df["mobile_class"] == "parts") & \
                                    (pd.isna(data_df["mobile_part_code"])), "url"].values
        for _url in mobile_url_list:
            mobile_part_name_list = repair_df.loc[repair_df["url"] == _url].sort_values(["repair_result", "merging_selected", "merging_rank"], ascending=False)["part_name"].values
            if len(mobile_part_name_list) > 0:
                part_name = mobile_part_name_list[0]
                part_code = PARTS_DF.loc[PARTS_DF["part_name"] == part_name, "part_code"].values.item()

                data_df.loc[(data_df["mobile_class"] == "parts") & (pd.isna(data_df["mobile_part_code"])), \
                            "mobile_part_name"] = part_name
                data_df.loc[(data_df["mobile_class"] == "parts") & (pd.isna(data_df["mobile_part_code"])), \
                            "mobile_part_code"] = part_code

    # Make output dict
    ## Make ai_result
    if len(repair_df) > 0 :
        _ai_result_df = repair_df.loc[repair_df["repair_result"] == True, \
                                      ["index", "file_index", "url", \
                                       "part_name", "part_score", "part_code", \
                                       "repair_class_name", "repair_score", "repair_code", \
                                       "damage-result-ratio", "damage-result-section", \
                                       "damage-result-list", "damage-result-code"]]
        _ai_result_df = _ai_result_df.sort_values("repair_code", ascending=False)
        _ai_result_df = _ai_result_df.round(4)
        _ai_result_df = _ai_result_df.fillna("")
        _ai_result_df.index = ["part_" + str(x) for x in range(len(_ai_result_df.index))]

        _ai_result_dict = _ai_result_df.transpose().to_dict()

    else :
        _ai_result_dict = {}

    ## Make ai_info
    _ai_info_dict = dict()
    for ai_info_idx, url_idx in enumerate(data_df["url"]) :
        _key = "file_" + str(ai_info_idx)

        _file_df = data_df.loc[data_df["url"] == url_idx, ["file_index", "url", "shape", "service_code", \
                                                           "mobile_class", "mobile_mode", "mobile_tiny", \
                                                           "mobile_part_code", "mobile_part_name", \
                                                           "aos_part_code", "aos_part_name", "aos_part_group"]]
        _file_df = _file_df.round(4)
        _file_df = _file_df.fillna("")
        _file_df.index = ["file"]

        _filter_df = data_df.loc[data_df["url"] == url_idx, ["filter_class", "filter_class_name", \
                                                             "filter_score", "filter_probs", "filter_code"]]
        _filter_df = _filter_df.round(4)
        _filter_df = _filter_df.fillna("")
        _filter_df.index = ["filter"]

        if len(repair_df) > 0 :
            _repair_df = repair_df.loc[repair_df["url"] == url_idx, \
                                       [x for x in repair_df.columns if all([y not in x for y in ["image", "mask", "url", "ref", "file_index"]])]]
            _repair_df = _repair_df.round(4)
            _repair_df = _repair_df.fillna("")
            _repair_df.index = ["part_" + str(x) for x in range(len(_repair_df.index))]
        else :
            _repair_df = pd.DataFrame([])

        _ai_info_dict[_key] = dict()
        _ai_info_dict[_key].update({"file" : list(_file_df.transpose().to_dict().values())})
        _ai_info_dict[_key].update({"filter" : list(_filter_df.transpose().to_dict().values())})
        _ai_info_dict[_key].update({"part" : list(_repair_df.transpose().to_dict().values())})

    ## Make etc
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

    return output_dict, data_df, repair_df

def gen_valid_image(data_df, repair_df) :

    if repair_df.shape[0] > 0 :
        report_df = repair_df.sort_values(by=["url", "merging_score"], ascending=False).reset_index(drop=True)
        report_df = report_df.loc[(report_df["repair_result"] == True) & (report_df["repair_class"] != 0)]

        report_df["valid_image"]=None
        report_df["valid_image_b64"]=None

        # display outputs
        ## selected
        image_dict = dict()
        for idx in report_df.index.to_numpy() :
            ### Get info
            _url = report_df.loc[report_df.index == idx, "url"].values[0]
            _part_name = report_df.loc[report_df.index == idx, "part_name"].values[0]

            ### Image handle
            _image = np.array(data_df.loc[data_df["url"] == _url, "image"].values[0], dtype=np.uint8)
            _damage_seg_map = np.array(report_df.loc[report_df.index == idx, "damage-summary-mask"].values[0])

            if len(_damage_seg_map.shape) == 0 :
                _damage_seg_map = np.zeros(list(_image.shape)[:2], dtype=np.uint8)
            elif len(_damage_seg_map.shape) == 3 :
                _damage_seg_map = _damage_seg_map[-1]

            _damage = segmap2image(np.clip(_damage_seg_map, 0, 1), DAMAGE_DF)
            _valid = apply_mask(_image, _damage)

            # Make base64
            _valid_image = Image.fromarray(_valid)

            buffer = io.BytesIO()
            _valid_image.save(buffer, format='jpeg')
            # buffer.seek(0)

            # _valid_image_b64 = base64.b64encode(buffer.read()).decode('utf-8')

            # Insert result
            # report_df.loc[report_df.index == idx, "valid_image_url"] = _url
            # report_df.loc[report_df.index == idx, "valid_image"] = buffer
            # report_df.loc[report_df.index == idx, "valid_image"] = [_valid]
            # report_df.loc[report_df.index == idx, "valid_image_b64"] = _valid_image_b64
            image_dict['file{}'.format(idx)] = ('file{}'.format(idx), buffer, 'image/jpg')
            # report_df.loc[report_df.index == idx, "valid_image"] = image_dict
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
  """Computes area of boxes.

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