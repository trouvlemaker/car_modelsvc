import time
from datetime import date
import requests
import cv2, json
import pickle, gzip
from pathlib import Path
import matplotlib.pyplot as plt
from pprint import pprint
import copy

from app.common.utils import *
from app.ai.tools_sds import *

from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing

from app.common.aos_exceptions import AosModelServiceError
from app.ai.config import *
import numpy as np

################ Damage ##############
from app.mod_service import models_sds as models
from PIL import Image
import io
from skimage.draw import polygon2mask
################ Damage ##############

logger = create_logger(__name__)
app_env = os.getenv('APP_ENV')


######################################
# 견적 프로세스 함수 
# 원거리 - (중거리/근접) 견적 프로세스 진행
######################################
def predict(pcs_id, pcs_sn, estReqDiv, data, dmg_pats, mode='predict'):
    # output = list()
    log_pcs = '{}|{}|{}'.format(pcs_id, pcs_sn, estReqDiv)

    print('[{}] Start Prediction. image count {}'.format(log_pcs, len(data)), flush=True)
    # 전체 견적 프로세스 total time
    # 원거리 견적 total time
    # 근거리 견적 total time
    # 이미지 매칭 total time
    total_predict = time.time()
    far_calc_time = 0
    close_calc_time = 0
    im_time = 0

    # start time check
    start = time.time()

    with ThreadPoolExecutor(max_workers=64) as pool:
        try:
            if mode == 'filter':
                output = list(pool.map(filter_inference_trtserving, data))             
            else:
                part_start = time.time()
                output = list(pool.map(inference_trtserving_far_image, data))

                end = time.time()
                far_calc_time = end - start

                if mode != 'filter' and mode != 'masking':
                    infer_time_max = 0
                    cur_infer_time = None
                    tmp_logs = list()
                    for o in output:
                        o_infer_time = o.get('inference_time', None)
                        if o_infer_time and o_infer_time['total'] > infer_time_max:
                            infer_time_max = o_infer_time['total']
                            cur_infer_time = o_infer_time

                    if cur_infer_time:
                        print('[{}] Max  {}'.format(log_pcs, cur_infer_time), flush=True)

                ##########################
                #근접 / 원거리 이미지 분류
                ##########################
                #근접 견적 프로세스 time check	
                close_start = time.time()
                #원거리 데이터
                far_img_list = []
                far_output = []
                far_rule_list= []
                #근거리 데이터
                close_img_list = []
                close_output = []
                close_rule_list = []

                close_output_list = []
               
                for _, report in enumerate(output):
                    max_part_area = 0.0
                    part_code_list = []
                    
                    if report['filter_res_out']['check']:
                        if (report['part_res_out']['check']):
                            height, width, _ = report['data_res_out']['shape']
                            imaeg_area = height*width

                            for part_nm in report['part_res_out']['report'].keys():
                                part_score = report['part_res_out']['report'][part_nm]['score']
                                
                                #부품 스코어가 0.8이상만 채택
                                if part_score >= 0.8:
                                    part_code_list.append(part_nm)
                                    each_part_area = np.sum(report['part_res_out']['report'][part_nm]['resized_mask'])/imaeg_area
                                    if max_part_area < each_part_area:
                                        max_part_area = each_part_area
                                
                            part_code_list = np.unique(np.asarray(part_code_list)).tolist()
                        
                        #원거리 - 근거리 룰셋
                        far_rule = far_close_rule(report, max_part_area, part_code_list)  
                        
                        #위 룰 중 하나라도 해당되면 원거리
                        if (far_rule > 0):
                            far_rule_list.append({"part_cnt": len(part_code_list), "url" : report['data_res_out']['url'], "image" : None , "report":report})
                        else:
                            close_rule_list.append({"part_cnt": len(part_code_list), "url" : report['data_res_out']['url'], "image" : None , "report":report}) 

            #견적 프로세스 전체 time check
            total_predict_end = time.time()
                
            if mode != 'masking':
                # report, data_df, repair_df = merge_output(output, dmg_pats)
                report, data_df = merge_output(output, dmg_pats)
                report['misc_info']['calc_time'] = far_calc_time + close_calc_time
                report['misc_info']['pcs_id'] = pcs_id
                report['misc_info']['pcs_sn'] = pcs_sn
                total_predict_time = total_predict_end - total_predict
                print(f"[{log_pcs}] Total Prediction completed: {round(total_predict_time, 2)}|{round(far_calc_time, 2)}, image count: {len(output)}", flush=True)
                report['results'] = 'success'
                return json.dumps(report, cls=NumpyEncoder, indent=4)
            else:
                print('[{}] Total Masking completed: {}, image count: {}'.format(log_pcs, round(far_calc_time, 2), len(output)),
                      flush=True)            
                return json.dumps(output, cls=NumpyEncoder, indent=4)
        except Exception as e:
            logger.exception('[{}] Prediction exception -> {}'.format(log_pcs, str(e)))
            e_report = dict()
            misc_info = {'pcs_id': pcs_id,
                         'pcs_sn': pcs_sn,
                         'exception': str(e)}
            e_report['misc_info'] = misc_info

            return json.dumps(e_report)


def filter_inference_trtserving(data):
    start_inference = time.time()

    #####################################################################
    # 'image_data': {'check': True, 'url': '/images/images_0914_PM/test/001/images/101-02-83235_20151106161_67오7643_418987.jpg', 'image': array([[[ 30,  21,  24]....., dtype=uint8), 'shape': (600, 800, 3)}
    #####################################################################
    input_data = data['image_data']

    aos_tag = None
    if 'aos_tag' in data.keys():
        aos_tag = data['aos_tag']

    aos_part_group = None
    if 'aos_part_group' in data.keys():
        aos_part_group = data['aos_part_group']

    # set params
    report = {"code": [],
              "data_res_out": input_data}

    inference_time = dict()

    ######################################################################
    # Check Mobile info
    try:
        mobile_info = mobile_info_parser(data['image_url'])
        report["mobile_info"] = mobile_info
    except:
        report["mobile_info"] = {"check": False}

    # Check AOS info
    try:
        aos_info = aos_info_parser(part_code=aos_tag,
                                   is_group=aos_part_group)  # aos_tag 변수명 확인
        report["aos_info"] = aos_info
    except:
        report["aos_info"] = {"check": False}

    ######################################################################
    # Filter model
    if report["data_res_out"]["check"]:
        try:
            report['filter_res_out'], inference_time['filter'] = models.filter_cls_detection(report, image_out=False)
        except Exception as e:
            raise AosModelServiceError(e)
    else:
        report['code'].append('11')

    if 'filter_res_out' not in report.keys():
        report['filter_res_out'] = {'check': False}

    ######################################################################
    # Fill-out dummy report
    if "part_res_out" not in report.keys():
        report["part_res_out"] = {"check": False}

    if "damage_res_out" not in report.keys():
        report["damage_res_out"] = {"check": False}

        
    ######################################################################
    # Summary
    if len(report["code"]) == 0:
        report["code"].append("00")  ## All Good.

    report = summarize(data=report)
   
    
    ######################################################################
    # Done

    stop_inference = time.time()
    logger.debug('Filter completed : {} : {}'.format(stop_inference - start_inference, data['image_url']))
    # image_name = input_data['url'].rsplit('/')[-1]
    # inference_time['total-{}'.format(image_name)] = round(stop_inference - start_inference, 2)
    # report['inference_time'] = inference_time
    # return report
    return report

#SDS Image Matching
def inference_trtserving_image_matching(far_img_list, close_img_list):
    result, inference_time  = models.image_matching(far_img_list, close_img_list)
    return result, inference_time
            
'''SDS Process'''
######################################
# 원거리 견적 프로세스 함수 
# 필터 - 부품 - 원거리 손상 - 전처리
# 1. 원거리 손상 Segmentation으로 변경
# 2. 전처리 이후 원거리 손상 영역과 합성 실시
######################################
def inference_trtserving_far_image(data):
    
    start_inference = time.time()

    input_data = data['image_data']

    aos_tag = None
    if 'aos_tag' in data.keys():
        aos_tag = data['aos_tag']

    aos_part_group = None
    if 'aos_part_group' in data.keys():
        aos_part_group = data['aos_part_group']

    # set params
    # image_matching 및 전처리 params 추가
    report = {"code": [],
              "data_res_out": input_data,
              "im_res_out" : {'far_image_url' : None, 'check' : False},
              "preprocess_res_out" :{'check': False}}

    report['tmp_quantile'] = np.quantile(input_data['image'], [0.25, 0.5, 0.75, 1])
    report['tmp_round'] = np.round(np.array([np.nanmean(input_data['image']), np.nanstd(input_data['image']), np.nansum(input_data['image'])]))

    inference_time = dict()
    ######################################################################
    # Check Mobile info
    try:
        mobile_info = mobile_info_parser(input_data['url'])
        report["mobile_info"] = mobile_info
        '''Debug'''
        #print("FAR_IMAGE = M-CHECK :", mobile_info["check"],"M-CLASS :", mobile_info["class"],"M-MODE :", mobile_info["mode"],"M-TINY :", mobile_info["tiny"],"M-PARTCODE :", mobile_info["part_code"],"M-PARTNAME :", mobile_info["part_name"])
        
    except:
        report["mobile_info"] = {"check" : False}

    # Check AOS info
    try:
        aos_info = aos_info_parser(part_code=aos_tag,
                                   is_group=aos_part_group)  # aos_tag 변수명 확인
        report["aos_info"] = aos_info
    except:
        report["aos_info"] = {"check": False}

    ######################################################################
    # Filter model
    if report["data_res_out"]["check"]:
        try:
            report['filter_res_out'], inference_time['filter'] = models.filter_cls_detection(report)
        except Exception as e:
            raise AosModelServiceError(e)
    else:
        report['code'].append('11')

    if 'filter_res_out' not in report.keys():
        report['filter_res_out'] = {'check': False}

    #####################################################################
    middle_time = time.time()

    # ######################################################################
    # Parts model
    if report["filter_res_out"]["check"]:
        try:
            # logger.debug('call parts model %s' % data['image_url'])
            # report['part_res_out'], inference_time['parts'] = models.parts_detection(report, masking=True, masking_output_path ="./output/{0}".format(report["data_res_out"]["url"].split('/')[-1]))
            report['part_res_out'], inference_time['parts'] = models.parts_detection(report)
            middle_time2 = time.time()
        except Exception as e:
            raise AosModelServiceError(e)
    else:
        if not any([x in ["11", "31"] for x in report["code"]]):
            report["code"].append("21")  ## Filter out.
            
    if 'part_res_out' not in report.keys():
        report['part_res_out'] = {'check': False}        
        ## image_matching 추가
        report['im_res_out'] = {'far_image_url' : None, 'check' : False}

    pre_total_time = 0.0 #sum(report['inference_time'] .values())
    
    # ######################################################################
    # Damage model
    if report["filter_res_out"]["check"]:
        try:
            damage_report, inference_time['damage'] = models.damage_detection(report)
            # damage_report, damage_seg_map, inference_time['damage'] = models.damage_detection(report)
            damage_check = False if len(damage_report) == 0 else True
            damage_res_out = {"check" : damage_check,
                               "report" : damage_report,
                            }
            report["damage_res_out"] = damage_res_out
            if report["damage_res_out"]["check"] == False :
                report["code"].append("23") ## No damage detected.

        except Exception as except_damage:
            logger.exception('Exception damage model : {} | {} | {}'.format(input_data['url'], type(except_damage), except_damage))
            raise AosModelServiceError(except_damage)
    else:
        report["code"].append("22")

    if "damage_res_out" not in report.keys():
        damage_res_out = {"check": False, 
                             "report" : None,
                             "seg_map" : None}
        report["damage_res_out"] = damage_res_out         
    
    ######################################################################
    # Map damage_res_out & part_res_out

    # report = damage_parts_mapper(data=report, img_path_name=report["data_res_out"]["url"].split('/')[-1])
    # report = damage_parts_mapper(data=report, img_path_name=report["data_res_out"]["url"].split('/')[-1], json_path_name=report["data_res_out"]["url"].split('/')[-1])
    report = damage_parts_mapper(data=report)

    # ######################################################################
    # Summary
    if len(report["code"]) == 0 :
        report["code"].append("00") ## All Good.

    report = summarize(data=report)

    ######################################################################
    # Done

    stop_inference = time.time()
    logger.debug('Prediction completed : {} : {}'.format(stop_inference - start_inference, input_data['url']))
    image_name = input_data['url'].rsplit('/')[-1]
    predict_total_time = sum(inference_time.values())
    total_time = round(stop_inference - start_inference, 2)
    inference_time['proc'] = round(total_time - predict_total_time, 2)
    inference_time['total'.format(image_name)] = total_time
    report['inference_time'] = inference_time

    return report


