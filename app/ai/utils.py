import re
import logging
import sys
import uuid
import traceback

import copy
from datetime import datetime

# from app.run import db
# from app.db_log.model_svc_log import SPSDModelSvcLog


def generate_uuid():
    return 'BM' + datetime.today().strftime('%m%d%S') + str(uuid.uuid4())[:8]


# class SQLAlchemyHandler(logging.Handler):
#     def emit(self, record):
#         trace = None
#         level = record.__dict__['levelname']
#         if level in ['ERROR']:
#             exc = record.__dict__['exc_info']
#             if exc:
#                 trace = traceback.format_exc()
#             timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#             e_msg = record.__dict__['msg']
#             p = re.compile('\[([\w\d]*)\|([\w\d]*)\|([\w\d]*)\]')
#             match = p.search(e_msg)
#             if match:
#                 log = SPSDModelSvcLog(logger=record.__dict__['name'],
#                                       level=level,
#                                       pcsid=match.group(1),
#                                       pcssn=match.group(2),
#                                       est=match.group(3),
#                                       trace=trace,
#                                       msg=record.__dict__['msg'],
#                                       timestamp=timestamp)
#             else:
#                 log = SPSDModelSvcLog(logger=record.__dict__['name'],
#                                       level=level,
#                                       trace=trace,
#                                       msg=record.__dict__['msg'],
#                                       timestamp=timestamp)
#             db.session.add(log)
#             db.session.commit()


def create_logger(name):
    # db_handler = SQLAlchemyHandler()
    # db_handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    # if not logger.handlers:
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    # logger.addHandler(db_handler)
    return logger


class TimeChecker:
    def __init__(self):
        self.last_point = datetime.now()

    def point(self):
        self.last_point = datetime.now()

    def check(self):
        temp_time = datetime.now()
        self.duration = temp_time - self.last_point
        self.last_point = temp_time

        return round(self.duration.total_seconds(), 2)
