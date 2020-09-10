from kafka_utils.producer import get_producer
from kafka_utils.consumer import get_consumer
import json
import tools.sp_enc_dec as sp
import ancillary_functions_anuvaad.ancillary_functions as ancillary_functions
import ancillary_functions_anuvaad.sc_preface_handler as sc_preface_handler
import ancillary_functions_anuvaad.handle_date_url as date_url_util
from config.config import statusCode,benchmark_types, language_supported, file_location
from config.kafka_topics import consumer_topics,producer_topics,kafka_topic
from onmt.utils.logging import init_logger,logger
import os
import datetime
from onmt.translate import ServerModelError
import sys

import translation_util.translate_util as translate_util


def doc_translator(translation_server,c_topic):
    logger.info('doc_translator')  
    iq =0
    out = {}
    msg_count = 0
    msg_sent = 0 
    c = get_consumer(c_topic)
    p = get_producer()
    try:
        for msg in c:
            producer_topic = [ topic["producer"] for topic in kafka_topic if topic["consumer"] == msg.topic][0]
            logger.info("Producer for current consumer:{} is-{}".format(msg.topic,producer_topic))
            msg_count +=1
            logger.info("*******************msg receive count*********:{}".format(msg_count))
            iq = iq +1
            inputs = (msg.value)

            if inputs is not None and all(v in inputs for v in ['url_end_point','message']) and len(inputs) is not 0:
                if inputs['url_end_point'] == 'translation_en':
                    logger.info("Running kafka on  {}".format(inputs['url_end_point']))
                    logger.info("Running kafka-translation on  {}".format(inputs['message']))
                    out = translate_util.from_en(inputs['message'], translation_server)
                elif inputs['url_end_point'] == 'translation_hi':
                    logger.info("Running kafka on  {}".format(inputs['url_end_point']))
                    logger.info("Running kafka-translation on  {}".format(inputs['message']))
                    out = translate_util.from_hindi(inputs['message'], translation_server)
                    logger.info("final output kafka-translation_hi:{}".format(out))  
                elif inputs['url_end_point'] == "translate-anuvaad":
                    logger.info("Running kafka on  {}".format(inputs['url_end_point']))
                    logger.info("Running kafka-translation on  {}".format(inputs['message']))  
                    out = translate_util.translate_func(inputs['message'], translation_server)
                    logger.info("final output kafka-translate-anuvaad:{}".format(out)) 
                else:
                    logger.info("Incorrect url_end_point for KAFKA")
                    out['status'] = statusCode["KAFKA_INVALID_REQUEST"]
                    out['response_body'] = []
            
            else:
                out = {}
                logger.info("Null input request or key parameter missing in KAFKA request: document_translator")       
          
            p.send(producer_topic, value={'out':out})
            p.flush()
            msg_sent += 1
            logger.info("*******************msg sent count*********:{}".format(msg_sent))
    except ValueError:  
        '''includes simplejson.decoder.JSONDecodeError '''
        logger.error("Decoding JSON has failed in document_translator: %s"% sys.exc_info()[0])
        doc_translator(translation_server,c_topic)  
    except Exception  as e:
        logger.error("Unexpected error in kafak doc_translator: %s"% sys.exc_info()[0])
        logger.error("error in doc_translator: {}".format(e))
        doc_translator(translation_server,c_topic)
