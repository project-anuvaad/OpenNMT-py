# -*- coding: utf-8 -*-
from __future__ import absolute_import

import logging
import colorama
import copy

logger = logging.getLogger()
LOG_TAGS = {"input":"[INPUT]","output":"[OUTPUT]"}

LOG_COLORS = {
    logging.ERROR: colorama.Fore.RED,
    logging.WARNING: colorama.Fore.YELLOW
}

'''
Using different color logs for console output
'''
class ColorFormatter(logging.Formatter):
    def format(self, record, *args, **kwargs):
        new_record = copy.copy(record)
        if new_record.levelno in LOG_COLORS:
            new_record.levelname = "{color_begin}{level}{color_end}".format(
                level=new_record.levelname,
                color_begin=LOG_COLORS[new_record.levelno],
                color_end=colorama.Style.RESET_ALL,
            )
        return super(ColorFormatter, self).format(new_record, *args, **kwargs)

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("%(asctime)s %(levelname)s %(filename)s %(lineno)d %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = ColorFormatter("%(asctime)s %(levelname)s %(filename)s %(lineno)d %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

def log_with_request_info(request_info_value,tag,message):
    '''
    request_info_value : request_id for document translation and s_id for interactive translation
    tag: INPUT or OUTPUT tag of the log
    '''
    final_log = "<"+str(request_info_value)+">" + " "+ tag + " " + str(message)
    return final_log

def entry_exit_log(tag,message):
    final_log = tag + " " + str(message)
    return final_log

def log_with_record_id(record_id,tag,message):
    '''
    record_id: record_id of request from WFM
    '''
    final_log = "{"+str(record_id)+"}" + " "+ tag + " " + str(message)
    return final_log