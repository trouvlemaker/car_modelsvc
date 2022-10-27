import os
import json
import logging
import sys
import functools
import yaml
from flask import Response, request
from stat import *
# from app import db
# from app.config import config
from functools import wraps
from pathlib import Path
from pprint import pprint



def create_logger(name):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    # if not logger.handlers:
    logger.propagate = False
    # if config['DEBUG']:
    #     logger.setLevel(logging.DEBUG)
    # else:
    #     logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


logger = create_logger(__name__)


# json parameter decorator
def required_params(required):
    def decorator(fn):

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _json = request.get_json()
            logger.info('<{}> function request parameters = {}'.format(wrapper.__name__, _json))
            missing = [r for r in required.keys()
                       if r not in _json]
            if missing:
                response = {
                    "success": False,
                    "log": "Request JSON is missing some required params. [{}]".format(missing)
                }
                return response
                # return jsonify(response), 400
            wrong_types = [r for r in required.keys()
                           if not isinstance(_json[r], required[r])]
            if wrong_types:
                param_types = {k: str(v) for k, v in required.items()}
                response = {
                    "success": False,
                    "log": "Data types in the request JSON doesn't match the required format. {}".format(param_types)
                }
                return response
                # return jsonify(response), 400
            return fn(*args, **kwargs)

        return wrapper

    return decorator


# def wrap_try(fn, *args):
#     try:
#         result = fn(*args)
#         db.session.commit()
#         result['success'] = True
#         result['log'] = 'Success'
#         return result
#     except Exception as e:
#         db.session.rollback()
#         raise e
#

def as_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        res = f(*args, **kwargs)
        res = json.dumps(res, ensure_ascii=False, default=str).encode('utf8')
        return Response(res, content_type='application/json; charset=utf-8')

    return decorated_function
