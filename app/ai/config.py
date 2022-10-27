import sys, os

import pandas as pd
import numpy as np

# external module
from app.ai.mrcnn import *

FILTER_DF = pd.DataFrame({"class" : list(range(6 + 2)),
                          "fast_pass": [False, False, False, True, True, True, True, True],
                          "mobile_class": [None, None, None, "vin", "dash", "fix", "ocr", "mobile_etc"]
                          },
                         index=["Normal", "documents", "id", "vin_code", "dash_board", "repair", "ocr", "mobile_etc"])

PARTS_DF = pd.DataFrame(np.array([\
 ['back_door', 'back_door', 1, 'DGPT53', 40000.0, [50, 50, 150], ['rear_stop_left', 'rear_stop_right', 'rear_lamp_left', 'rear_lamp_right'], 4, 5, "REAR_BUMPER", "BUMPER", "SHEETING"],
 ['front_bumper', 'front_bumper', 2, 'DGPT11', 40000.0, [250, 50, 250], ['front_lamp_left', 'front_lamp_right'], 3, 3, "FRONT_BUMPER", "BUMPER", None],
 ['front_door_left', 'front_door', 3, 'DGPT41', 50000.0, [250, 250, 50], None, None, 5, "DOOR", "PANEL", "SHEETING"],
 ['front_door_right', 'front_door', 4, 'DGPT42', 50000.0, [150, 150, 250], None, None, 5, "DOOR", "PANEL", "SHEETING"],
 ['front_fender_left', 'front_fender', 5, 'DGPT31', 50000.0, [250, 50, 150], None, None, 5, "FENDER", "PANEL", "SHEETING"],
 ['front_fender_right', 'front_fender', 6, 'DGPT32', 50000.0, [250, 150, 250], None, None, 5, "FENDER", "PANEL", "SHEETING"],
 ['front_fog_left', None, 7, 'DGPT25', 10000.0, [150, 50, 150], None, None, None, None, None, None],
 ['front_fog_right', None, 8, 'DGPT26', 10000.0, [150, 150, 150], None, None, None, None, None, None],
 ['front_lamp_left', 'front_lamp', 9, 'DGPT21', 10000.0, [50, 50, 250], None, None, 2, "LAMP", "LAMP", None],
 ['front_lamp_right', 'front_lamp', 10, 'DGPT22', 10000.0, [250, 150, 150], None, None, 2, "LAMP", "LAMP", None],
 ['grille_up', 'grille_up', 11, 'DGPT13', 30000.0, [250, 250, 150], None, None, 2, "FRONT_BUMPER", "BUMPER", None],
 ['hood', 'hood', 12, 'DGPT51', 40000.0, [250, 250, 250], None, None, 5, "FRONT_BUMPER", "BUMPER", "SHEETING"],
 ['rear_bumper', 'rear_bumper', 13, 'DGPT12', 40000.0, [250, 50, 50], ['rear_lamp_left', 'rear_lamp_right', 'rear_stop_left', 'rear_stop_right'], 3, 3, "REAR_BUMPER", "BUMPER", None],
 ['rear_door_left', 'rear_door', 14, 'DGPT43', 50000.0, [150, 150, 50], None, None, 5, "DOOR", "PANEL", "SHEETING"],
 ['rear_door_right', 'rear_door', 15, 'DGPT44', 50000.0, [50, 250, 250], None, None, 5, "DOOR", "PANEL", "SHEETING"],
 ['rear_fender_left', 'rear_fender', 16, 'DGPT33', 50000.0, [150, 50, 50], None, None, 5, "FENDER", "PANEL", "SHEETING"],
 ['rear_fender_right', 'rear_fender', 17, 'DGPT34', 50000.0, [150, 250, 150], None, None, 5, "FENDER", "PANEL", "SHEETING"],
 ['rear_lamp_left', 'rear_lamp', 18, 'DGPT23', 10000.0, [50, 50, 50], None, None, 2, "LAMP", "LAMP", None],
 ['rear_lamp_right', 'rear_lamp', 19, 'DGPT24', 10000.0, [50, 150, 50], None, None, 2, "LAMP", "LAMP", None],
 ['rear_stop_center', 'rear_lamp', 20, 'DGPT28', 10000.0, [50, 150, 150], None, None, 2, None, None, None],
 ['rear_stop_left', 'rear_lamp', 21, 'DGPT27', 10000.0, [50, 250, 50], None, None, 2, None, None, None],
 ['rear_stop_right', 'rear_lamp', 22, 'DGPT29', 10000.0, [250, 150, 50], None, None, 2, None, None, None],
 ['side_mirror_left', None, 23, 'DGPT45', 10000.0, [150, 50, 250], None, None, 2, "SIDE_MIRROR", "SIDE_MIRROR", None],
 ['side_mirror_right', None, 24, 'DGPT46', 10000.0, [150, 250, 50], None, None, 2, "SIDE_MIRROR", "SIDE_MIRROR", None],
 ['side_step_left', "side_step", 25, 'DGPT61', 5000.0, [50, 150, 250], None, None, 2, "SIDE_STEP", "SIDE_STEP", None],
 ['side_step_right', "side_step", 26, 'DGPT62', 5000.0, [150, 250, 250], None, None, 2, "SIDE_STEP", "SIDE_STEP", None],
 ['trunk', 'trunk', 27, 'DGPT52', 40000.0, [50, 250, 150], ['rear_stop_left', 'rear_stop_right', 'rear_lamp_left', 'rear_lamp_right'], 4, 5, "REAR_BUMPER", "BUMPER", "SHEETING"],
 ['znumber_plate', None, 28, None, None, [200, 200, 50], None, None, None, None, None, None]]),
 index = ['back_door','front_bumper','front_door_left','front_door_right','front_fender_left','front_fender_right','front_fog_left','front_fog_right','front_lamp_left','front_lamp_right','grille_up','hood','rear_bumper','rear_door_left','rear_door_right','rear_fender_left','rear_fender_right','rear_lamp_left','rear_lamp_right','rear_stop_center','rear_stop_left','rear_stop_right','side_mirror_left','side_mirror_right','side_step_left','side_step_right','trunk', "znumber_plate"],
 columns = ['part_name', 'model_name', 'model_cls', 'part_code', 'min_size', 'rgb', 'section', 'section_len', "repair_cls_len", "part_group", "part_group_upper", "part_sheeting"]
 )

DAMAGE_DF = pd.DataFrame(np.array([\
 ['BG', 0, [0, 0, 0], 0, False, ""],
 ['scratch', 1, [0, 255, 0], 1, True, "DGCL01"],
 ['joint', 2, [255, 255, 0], 0, False, "DGCL04"],
 ['dent', 3, [0, 0, 255], 2, True, "DGCL02"],
 ['dent_press', 4, [255, 50, 50], 3, True, "DGCL02"],
 ['complex_damage', 5, [255, 50, 0], 9, True, "DGCL06"],
 ['broken', 6, [255, 150, 0], 8, True, "DGCL05"],
 ['removed', 7, [255, 0, 255], 0, False, ""]]),
 index = ['BG', 'scratch', 'joint', 'dent', 'dent_press', 'complex_damage', 'broken', 'removed'],
 columns = ["class_name", 'class', 'rgb', 'order', 'for_dnn', "code"]
 )
DAMAGE_DF = DAMAGE_DF.astype({"class": int, "order": int})

REPAIR_DF = pd.DataFrame(np.array([\
 ['RPCL00', 'intact', '정상'],
 ['RPCL12', 'painting', '스크래치'],
 ['RPCL21', 'repair', '수리'],
 ['RPCL31', 'sheeting_S', '판금(소)'],
 ['RPCL32', 'sheeting_M', '판금(중)'],
 ['RPCL33', 'sheeting_L', '판금(대)'],
 ['RPCL99', 'replace', '교환']]),
 columns = ["code", "class_name", "class_name_kr"]
 )

# Parts model config
class InferenceConfig(Config):
    NAME = "car"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    LEARNING_RATE = 1e-05

    MEAN_PIXEL = np.array([118, 118, 118])

    BACKBONE = "resnet50"
    NUM_CLASSES = PARTS_DF.shape[0] + 1
    IMAGE_MIN_DIM = 64 * 13
    IMAGE_MAX_DIM = 64 * 13
    RPN_ANCHOR_SCALES = (32 * 2, 32 * 5, 32 * 7, 32 * 11, 32 * 15)
    RPN_ANCHOR_RARIOS = [0.57446809, 1.1123348, 1.8627197]
    USE_MINI_MASK = False

    DETECTION_MIN_CONFIDENCE = 0.3
    DETECTION_NMS_THRESHOLD = 0.3

# Mask model inference configuration
MASK_CONFIG = InferenceConfig()


server_info = dict()
server_info['parts_model']='aosprj-parts-v1:9000'
# server_info['parts_model']='ambassador:80/modelsvc/tfserv-parts-model-v2'
server_info['damage_model']='aosprj-damage-v1:9000'
server_info['inception_model']='inference-backdoor-model-v1:9000'
# server_info='aos.devel.io:30011'
grpc_options=[('grpc.max_message_length', -1),
      ('grpc.max_send_message_length', -1),
      ('grpc.max_receive_message_length', -1)
]
REQ_TM_OUT=30
INFERENCE_TIMEOUT=50
REQ_TIMEOUT=12
CNN_REQ_TIMEOUT=1
DNN_REQ_TIMEOUT=3

model_option = {"parts"  : {"score_thres" : 0.75}, 
                   "damage" : {"score_thres" : 0.45,
                               "box_thres" : 0.1,
                               "TEST_SHORT_EDGE_SIZE" : 800,
                               "MAX_SIZE": 1333
                              },

                   "filter" : {"shape" : 224},

                   "cnn"    : {"shape" : 456,
                               "score_thres": 0}
                  }

# model_option = {"damage" : {"score_thres" : 0.45,
#                                "box_thres" : 0.1,
#                                "TEST_SHORT_EDGE_SIZE" : 800,
#                                "MAX_SIZE": 1333
#                               },
#                    "filter" : {"shape" : 224},
#                    "cnn"    : {"shape" : 456,
#                                 "score_thres": 0}
#                   }

'''SDS Config variable '''
#SDS Config 추가
# 근접 이미지매칭 사용 division ratio
SDS_IM_DIVIDE_RATIO = 2
# 근접 이미지매칭 사용 이미지 shape
SDS_CONFIG_IM_WIDTH = 800
SDS_CONFIG_IM_HEIGHT = 600

# 수리유형모델 결과 probs length 값
SDS_CONFIG_REPAIR_CLS_LEN = {
                'front_door_left' : 6,
                'front_door_right' : 6,
                'front_fender_left' : 6,
                'front_fender_right' : 6,
                'rear_door_left' : 6,
                'rear_door_right' : 6,
                'rear_fender_left' : 6,
                'rear_fender_right' : 6,
                'back_door' : 6
            }

# 근접 이미지매칭 허용 threshold
SDS_CONFIG_IM_SCORE = 0.25
# 근접에서 사용하는 부품 최소 score
SDS_CONFIG_CLOSE_PART_SCORE = 0.5
# 근접에서 사용하는 부품과 교차하는 손상영역 비율
SDS_CONFIG_CLOSE_DAMAGE_INTERSECT_RATIO = 0.05

#근접 견적 비즈니스 룰
# sheeting - dent 40% 이상시 교환
SDS_CONFIG_SHEETING_DENT_AREA_REPLACE_RATIO = 0.4
# sheeting - dent 25% 이상시 판금대
SDS_CONFIG_SHEETING_DENT_AREA_SHEETING_L_RATIO = 0.25
# sheeting - dent 15% 이상시 판금중
SDS_CONFIG_SHEETING_DENT_AREA_SHEETING_M_RATIO = 0.15
# bumper - dent 30% 이상시 교환
SDS_CONFIG_BUMPER_DENT_AREA_RATIO = 0.3
