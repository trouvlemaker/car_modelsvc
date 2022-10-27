import pandas as pd

# external module
from app.ai.mrcnn import *

PARTS_DF = pd.DataFrame({"model_name": ['back_door', 'front_bumper', 'front_door', 'front_door', 'front_fender', 'front_fender', None, None, 'front_lamp', 'front_lamp', 'grille_up', 'hood', 'rear_bumper', 'rear_door', 'rear_door', 'rear_fender', 'rear_fender', 'rear_lamp', 'rear_lamp', 'rear_lamp', 'rear_lamp', 'rear_lamp', None, None, None, None, 'trunk', None],
                         "model_cls": list(range(1, 29)),
                         "part_code": ['53', '11', '41', '42', '31', '32', '25', '26', '21', '22', '13', '51', '12', '43', '44', '33', '34', '23', '24', '28', '27', '29', '45', '46', '61', '62', '52', None], 
                         "min_size" : [40000, 40000, 50000, 50000, 50000, 50000, 10000, 10000, 10000, 10000, 30000, 40000, 40000, 50000, 50000, 50000, 50000, 10000, 10000, 10000, 10000, 10000, 10000, 10000, 5000, 5000, 40000, None],
                         "rgb" : [[50, 50, 150], [250, 50, 250], [250, 250, 50], [150, 150, 250], [250, 50, 150], [250, 150, 250], [150, 50, 150], [150, 150, 150], [50, 50, 250], [250, 150, 150], [250, 250, 150], [250, 250, 250], [250, 50, 50], [150, 150, 50], [50, 250, 250], [150, 50, 50], [150, 250, 150], [50, 50, 50], [50, 150, 50], [50, 150, 150], [50, 250, 50], [250, 150, 50], [150, 50, 250], [150, 250, 50], [50, 150, 250], [150, 250, 250], [50, 250, 150], [200, 200, 50]],
                         "section": [["rear_stop_left", "rear_stop_right", "rear_lamp_left", "rear_lamp_right"],
                                     ["front_lamp_left", "front_lamp_right"],
                                     ["damage_loc"],
                                     ["damage_loc"],
                                     ["front_lamp_left", "front_lamp_right", "front_door_left", "front_door_right", "front_bumper"],
                                     ["front_lamp_left", "front_lamp_right", "front_door_left", "front_door_right", "front_bumper"],
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     ["front_lamp_left", "front_lamp_right"],
                                     ["rear_lamp_left", "rear_lamp_right", "rear_stop_left", "rear_stop_right"],
                                     ["damage_loc"],
                                     ["damage_loc"],
                                     ["rear_bumper"],
                                     ["rear_bumper"],
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     None,
                                     ["front_door_left", "rear_door_left"],
                                     ["front_door_right", "rear_door_right"],
                                     ["rear_stop_left", "rear_stop_right", "rear_lamp_left", "rear_lamp_right"],
                                     None],
                         "section_len" : [4, 3, 4, 4, 4, 4, None, None, None, None, None, 2, 3, 4, 4, 2, 2, None, None, None, None, None, None, None, 3, 3, 4, None]
                         },
                        index = ['back_door','front_bumper','front_door_left','front_door_right','front_fender_left','front_fender_right','front_fog_left','front_fog_right','front_lamp_left','front_lamp_right','grille_up','hood','rear_bumper','rear_door_left','rear_door_right','rear_fender_left','rear_fender_right','rear_lamp_left','rear_lamp_right','rear_stop_center','rear_stop_left','rear_stop_right','side_mirror_left','side_mirror_right','side_step_left','side_step_right','trunk', "number_plate"])


DAMAGE_DF = pd.DataFrame({"model_cls" : list(range(7)),
                          "rgb" : [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]}, 
                          index = ["background", "piercing", "scratch", "dent", "joint", "detach", "etc"])
DAMAGE_DF["min_size"] = 400

REPAIR_DF = {"00" : "정상",
             "11" : "스크래치",
             "21" : "수리",
             "31" : "판금(소)",
             "32" : "판금(중)",
             "33" : "판금(대)",
             "99" : "교환"}


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
    USE_MINI_MASK = False

    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3


# Mask model inference configuration
MASK_CONFIG = InferenceConfig()

server_info='tfserv-parts-model-v1:9000'
# server_info='aos.devel.io:30011'
grpc_options=[('grpc.max_message_length', -1),
      ('grpc.max_send_message_length', -1),
      ('grpc.max_receive_message_length', -1)
]
REQ_TM_OUT=10