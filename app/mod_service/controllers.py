# Import flask dependencies
import shutil
import uuid
import random
from datetime import datetime
import json, os
from pathlib import Path
from pprint import pprint
from flask import Blueprint, request, jsonify
from requests_toolbelt import MultipartEncoder
import cv2
import numpy as np
import zipfile
from glob import glob
import socket


from app.ai import tools_sds as tools
from app.ai import utils
from . import services_sds as services
from app.config import config
import gc

from app.common.utils import as_json, create_logger, required_params


logger = utils.create_logger(__name__)

if config['MODE'] == 'dev':
    mod_service = Blueprint('gateway', __name__, url_prefix='/dev/model')
elif config['MODE'] == 'single':
    mod_service = Blueprint('gateway', __name__, url_prefix='/singleway/model')
else:
    mod_service = Blueprint('gateway', __name__, url_prefix='/api')
# hm = memchecker.HeapMon()
# mem_tracker = tracker.SummaryTracker()

################################################################################
# Set the route and accepted methods
def generate_uuid():
    return 'M' + datetime.today().strftime('%m%d%S') + str(uuid.uuid4())[:8]

################################################################################



################################################################################
# 0928 filter, parts, damage model work process
@mod_service.route('/predict/', methods=['POST'])
def predict_model():

    ###########################################
    '''input data format
    test_json = {'pcs_id': 'test',
                'pcs_sn': "001",
                'mode': 'filter',
                'jsonData': json_files}

    add testMode local - inference server를 거치지 않고 로컬에서 데이터 수집

    json_files = [{'fileContentIndex': 'file0', 'fileId': '101-02-83235_20151106161_67오7643_418987.jpg'}, 
                {'fileContentIndex': 'file1', 'fileId': '102-06-10878_20170813187_9613_606108.jpg'}... ]
    '''
    ###########################################

    ret = None
    try:
        json_data = json.loads(request.form['jsonData'])
        svc_request = list()
        pcs_id = json_data['pcs_id']
        pcs_sn = json_data['pcs_sn']
        estReqDiv = json_data.get('estReqDiv', None)
        mode = json_data.get('mode', None)
        base_path = '{}'.format(config['IMAGE_BASE_PATH'])

        dmg_pats = list()
        if 'workItemCdData' in json_data.keys():
            for cd_data in json_data['workItemCdData']:
                dmg_pats.append(cd_data['aiDmgPatDivCd'])
        if config['MODE'] == 'dev' or config['MODE'] == 'test':
            print('--------------------------')
            pprint(dmg_pats)

        if 'jsonData' not in json_data.keys():
            logger.exception('jsonData not found {}|{}'.format(pcs_id, pcs_sn))
            e_report = dict()
            misc_info = {'pcs_id': pcs_id,
                         'pcs_sn': pcs_sn,
                         'exception': 'jsonData not found.'}
            e_report['misc_info'] = misc_info
            return json.dumps(e_report)

        if mode and mode == 'zip':
            transid = generate_uuid()
            base_path = '{}/{}'.format(config['IMAGE_BASE_PATH'], transid)

            data = json_data['jsonData'][0]
            file_content_id = data['fileContentIndex']
            file_name = data['fileId']
            file = request.files[file_content_id]

            image_path = '{}/{}/{}'.format(base_path, pcs_id, pcs_sn)
            image_zip_url = '{}/{}/{}/{}'.format(base_path, pcs_id, pcs_sn, file_name)

            p = Path(image_path)
            if not p.exists():
                os.makedirs(image_path)
            file.save(image_zip_url)
            del(file)

            with zipfile.ZipFile(image_zip_url, 'r') as zip_ref:
                zip_ref.extractall(image_path)
                for f in zip_ref.namelist():
                    image_url = '{}/{}/{}/{}'.format(base_path, pcs_id, pcs_sn, os.path.basename(f))

                    svc_image_data = dict()
                    image_data = tools.get_image(image_url)

                    svc_image_data['image_path'] = image_path
                    svc_image_data['image_url'] = image_url
                    svc_image_data['image_data'] = image_data
                    svc_request.append(svc_image_data)

            os.remove(image_zip_url)

            if base_path:
                p = Path(base_path)
                if p.exists():
                    shutil.rmtree(base_path)

            if mode:
                ret = services.predict(pcs_id, pcs_sn, estReqDiv, svc_request, dmg_pats, mode)
            else:
                ret = services.predict(pcs_id, pcs_sn, estReqDiv, svc_request, dmg_pats)
        else:
            today = 'images_{}'.format(datetime.today().strftime('%m%d_%p'))

            ##############################################
            # 임시코드 - request 이미지 저장
            # n = random.randint(1, 100)
            ##############################################

            for data in json_data['jsonData']:
                svc_image_data = dict()

                file_content_id = data['fileContentIndex']
                file_name = data['fileId']
                # file_data = data['image_data']
                # file_url = data['image_path']
                file = request.files[file_content_id]

                # [2020/03/05 - aosTag 추가]
                aosTag = None
                if 'aosTag' in data.keys():
                    aosTag = data['aosTag']
                    if config['MODE'] == 'dev' or config['MODE'] == 'test':
                        logger.info('aosTag value -> {}, image -> {}'.format(aosTag, file_name))

                aosPartGroup = None
                if 'aosPartGroup' in data.keys():
                    aosPartGroup = True if data['aosPartGroup'] == 'true' else False
                    if config['MODE'] == 'dev' or config['MODE'] == 'test':
                        logger.info('aosPartGroup value -> {}, image -> {}'.format(aosPartGroup, file_name))

                if file_name is None:
                    logger.error('Exception raised in controller.[image file name not found in jsonData.]')
                    e_report = dict()
                    misc_info = {'pcs_id': pcs_id,
                                 'pcs_sn': pcs_sn,
                                 'exception': 'image file name not found in jsonData.'}
                    e_report['misc_info'] = misc_info
                    return json.dumps(e_report)

                # fid = generate_uuid()

                image_path = '{}/{}/{}/{}/images'.format(base_path, today, pcs_id, pcs_sn)
                image_url = '{}/{}/{}/{}/images/{}'.format(base_path, today, pcs_id, pcs_sn, file_name)
                image_data = _get_image_from_stream(file.stream, file_name)
                # image_data = _get_image_from_online(file_url,file_data)
                del(file.stream)

                # check file
                if not image_data['check']:
                    logger.error('Exception raised in controller.[image file not found.]')
                    e_report = dict()
                    misc_info = {'pcs_id': pcs_id,
                                 'pcs_sn': pcs_sn,
                                 'exception': 'image file not found.'}
                    e_report['misc_info'] = misc_info
                    return json.dumps(e_report)

                #############################################
                # 임시코드 - request 이미지 저장
                # if config['MODE'] == 'dev':
                #     p = Path(image_path)
                #     if not p.exists():
                #         os.makedirs(image_path)
                #     tmp_image_url = image_url.replace('jpg', 'png')
                #     r_image = image_data['image']
                #     r_image = cv2.cvtColor(r_image, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(tmp_image_url, r_image)
                #############################################

                image_data['url'] = image_url
                svc_image_data['image_path'] = image_path
                svc_image_data['image_url'] = image_url
                svc_image_data['image_data'] = image_data
                svc_image_data['aos_tag'] = aosTag
                svc_image_data['aos_part_group'] = aosPartGroup
                svc_request.append(svc_image_data)
                
                ###########################################
                '''svc_request data format
                [{
                    'image_path': '/images/images_0914_PM/test/001/images', 
                    'image_url': '/images/images_0914_PM/test/001/images/101-02-83235_20151106161_67오7643_418987.jpg', 
                    'image_data': {'check': True, 'url': '/images/images_0914_PM/test/001/images/101-02-83235_20151106161_67오7643_418987.jpg', 'image': array([[[ 30,  21,  24]....., dtype=uint8), 
                    'shape': (600, 800, 3)}, 
                    'aos_tag': None, 'aos_part_group': None
                },...]
                '''
                ###########################################

            if mode:
                ret = services.predict(pcs_id, pcs_sn, estReqDiv, svc_request, dmg_pats, mode)
            else:
                ret = services.predict(pcs_id, pcs_sn, estReqDiv, svc_request, dmg_pats)

            ##############################################
            # 임시코드 - response JSON 저장
            # if config['MODE'] == 'dev' or config['MODE'] == 'test':
            #     output_file = '{}/tmp_json/{}-{}-{}-results.json'.format(base_path, today, pcs_id, pcs_sn)
            #     with open(output_file, 'w') as f:
            #         f.write(ret)
            ##############################################

        logger.debug('All Done.')

        # gc.collect()
        # print(hm.check('After predict'), flush=True)

        return ret
    except Exception as e:
        # logger.error('Exception raised in controller.[%s]' % e)
        e_resp = {'misc_info': {'exception': str(e)}}
        return jsonify(e_resp)


@mod_service.route('/predict/save/', methods=['POST'])
def predict_model_image_save():
    ret = None

    try:
        json_data = json.loads(request.form['jsonData'])

        svc_request = list()
        pcs_id = json_data['pcs_id']
        pcs_sn = json_data['pcs_sn']
        estReqDiv = json_data.get('estReqDiv', None)
        mode = json_data.get('mode', None)
        option = json_data.get('option', None)
        # transid = generate_uuid()
        # base_path = '{}/{}'.format(config['IMAGE_BASE_PATH'], transid)
        base_path = '{}'.format(config['IMAGE_BASE_PATH'])

        dmg_pats = list()
        if 'workItemCdData' in json_data.keys():
            for cd_data in json_data['workItemCdData']:
                dmg_pats.append(cd_data['aiDmgPatDivCd'])

        if 'jsonData' not in json_data.keys():
            logger.exception('jsonData not found {}|{}'.format(pcs_id, pcs_sn))
            e_report = dict()
            misc_info = {'pcs_id': pcs_id,
                         'pcs_sn': pcs_sn,
                         'exception': 'jsonData not found.'}
            e_report['misc_info'] = misc_info
            return json.dumps(e_report)

        if mode and mode == 'zip':
            transid = generate_uuid()
            base_path = '{}/{}'.format(config['IMAGE_BASE_PATH'], transid)

            data = json_data['jsonData'][0]
            file_content_id = data['fileContentIndex']
            file_name = data['fileId']
            file = request.files[file_content_id]

            image_path = '{}/{}/{}'.format(base_path, pcs_id, pcs_sn)
            image_zip_url = '{}/{}/{}/{}'.format(base_path, pcs_id, pcs_sn, file_name)

            p = Path(image_path)
            if not p.exists():
                os.makedirs(image_path)
            file.save(image_zip_url)

            with zipfile.ZipFile(image_zip_url, 'r') as zip_ref:
                zip_ref.extractall(image_path)
                for f in zip_ref.namelist():
                    image_url = '{}/{}/{}/{}'.format(base_path, pcs_id, pcs_sn, os.path.basename(f))

                    svc_image_data = dict()
                    image_data = tools.get_image(image_url)

                    svc_image_data['image_path'] = image_path
                    svc_image_data['image_url'] = image_url
                    svc_image_data['image_data'] = image_data
                    svc_request.append(svc_image_data)

            os.remove(image_zip_url)

            if base_path:
                p = Path(base_path)
                if p.exists():
                    shutil.rmtree(base_path)

            if option:
                ret = services.predict(pcs_id, pcs_sn, estReqDiv, svc_request, dmg_pats, option)
            else:
                ret = services.predict(pcs_id, pcs_sn, estReqDiv, svc_request, dmg_pats)
        else:
            today = 'images_{}'.format(datetime.today().strftime('%m%d_%p'))

            ##############################################
            # 임시코드 - request 이미지 저장
            n = random.randint(1, 100)
            # n = 5
            ##############################################

            for data in json_data['jsonData']:
                svc_image_data = dict()

                file_content_id = data['fileContentIndex']
                file_name = data['fileId']
                file = request.files[file_content_id]

                if file_name is None:
                    logger.error('Exception raised in controller.[image file name not found in jsonData.]')
                    e_report = dict()
                    misc_info = {'pcs_id': pcs_id,
                                 'pcs_sn': pcs_sn,
                                 'exception': 'image file name not found in jsonData.'}
                    e_report['misc_info'] = misc_info
                    return json.dumps(e_report)

                # fid = generate_uuid()

                image_path = '{}/{}/{}/{}/images'.format(base_path, today, pcs_id, pcs_sn)
                image_url = '{}/{}/{}/{}/images/{}'.format(base_path, today, pcs_id, pcs_sn, file_name)
                # image_data = _get_image_from_stream(file, file_name)
                p = Path(image_path)
                if not p.exists():
                    # logger.debug('Create directory for image download [%s]' % image_path)
                    os.makedirs(image_path)
                logger.info("Save file [%s]" % image_url)
                file.save(image_url)
                image_data = tools.get_image(image_url)

                # check file
                if not image_data['check']:
                    logger.error('Exception raised in controller.[image file not found.]')
                    e_report = dict()
                    misc_info = {'pcs_id': pcs_id,
                                 'pcs_sn': pcs_sn,
                                 'exception': 'image file not found.'}
                    e_report['misc_info'] = misc_info
                    return json.dumps(e_report)

                image_data['url'] = image_url
                svc_image_data['image_path'] = image_path
                svc_image_data['image_url'] = image_url
                svc_image_data['image_data'] = image_data
                svc_request.append(svc_image_data)

            if option:
                ret = services.predict(pcs_id, pcs_sn, estReqDiv, svc_request, dmg_pats, option)
            else:
                ret = services.predict(pcs_id, pcs_sn, estReqDiv, svc_request, dmg_pats)

        if image_path:
            p = Path(image_path)
            if p.exists():
                shutil.rmtree(image_path)

        logger.debug('All Done.')

        return ret
    except Exception as e:
        # logger.error('Exception raised in controller.[%s]' % e)
        e_resp = {'misc_info': {'exception': str(e)}}
        return jsonify(e_resp)



def _get_image_from_stream(f, f_name):
    result_dict = dict()
    try:
        check = True

        image = cv2.imdecode(np.fromstring(f.read(), np.uint8), cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        o_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = None
    except Exception as e:
        logger.error('Exception when parsing image -> %s' % e)
        image = None
        check = False

    result_dict['check'] = check
    result_dict['url'] = f_name
    result_dict['image'] = o_image
    result_dict['shape'] = o_image.shape if o_image is not None else None
    return result_dict

def _get_image_from_online(url,data):
    result_dict = dict()
    try:
        check = True
        image = np.array(data, dtype=np.uint8)
        o_image = cv2.imdecode(image, cv2.COLOR_BGR2RGB)
        image = None
    except Exception as e:
        logger.error('Exception when parsing image -> %s' % e)
        image = None
        check = False

    result_dict['check'] = check
    result_dict['url'] = url
    result_dict['image'] = o_image
    result_dict['shape'] = o_image.shape if o_image is not None else None
    return result_dict