import os
import json
import requests
from glob import glob 
from datetime import datetime
import cv2
import numpy as np
import base64


class PredictTest():

    def __init__(self, test_image_path, url='localhost', port='8090'):
        self.gateway_url = f"http://{url}:{port}/api/predict/"
        self.images_path = test_image_path

        file_list = glob(os.path.join(self.images_path, '*'))
        json_files = [{'fileContentIndex': 'file{}'.format(idx), 'fileId': os.path.basename(f)}
                      for idx, f in enumerate(file_list)]

        self.files_data = {'file{}'.format(idx): (os.path.basename(f), open(f, 'rb'), 'application/octet-stream') if (os.path.basename(f).split(".")[-1] !='zip') else (os.path.basename(f), open(f, 'rb'), 'application/zip')
                      for idx, f in enumerate(file_list)}      

        test_json = {'pcs_id': 'test',
                     'pcs_sn': "001",
                    #  'mode': 'image',
                     'mode': 'zip',
                     'jsonData': json_files,
                     }
        print(json_files)
        self.files_data['jsonData'] = (None, json.dumps(test_json), 'application/json')


    def predict(self, save_json_dir=None):

        def save_json(config, config_path):
            """
            dict 형태의 데이터를 받아 json파일로 저장해주는 함수
            """
            now = datetime.now()

            result_output_dir='{}'.format(config_path)
            json_dir = os.path.join(result_output_dir, "json")
            os.makedirs(json_dir, exist_ok=True)
            save_path = os.path.join(json_dir, str(now.date())+ "_"+ str(int(round(now.timestamp(),0)))+".json")
                    
            def default(obj):
                if type(obj).__module__ == np.__name__:
                    if isinstance(obj, np.ndarray):
                        # return obj.tolist()
                        return []
                    else:
                        return []
                # raise TypeError('Unknown type:', type(obj))

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4, default=default)

        r = requests.request('POST', self.gateway_url, files=self.files_data)
        if r.status_code == 200:
            rest = json.loads(r.content)
            if save_json_dir:
                save_json(rest, save_json_dir)
            # print(rest)
            if 'results' in rest.keys() and rest['results'] == 'success':
                print('test ok')
            else:
                r.raise_for_status()
        else:
            print('Failed prediction request ' + str(r.status_code))
            r.raise_for_status()

if __name__ == "__main__":
    # image_path = './sample_images/'
    # image_path = './damage_test/'
    # image_path = './test_image/'
    image_path = './test_image/test_zip/'
    # image_path = './car_images/sample_images/'

    tester = PredictTest(image_path)
    tester.predict(save_json_dir="./output/result_json/")
