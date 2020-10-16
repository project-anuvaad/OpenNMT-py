import json
from kafka import KafkaProducer
from onmt.utils.logging import init_logger,logger
import os

KAFKA_IP_HOST = 'KAFKA_IP_HOST'
default_value = 'localhost:9092'
bootstrap_server = os.environ.get(KAFKA_IP_HOST,default_value)


def get_producer():
    try:
        producer = KafkaProducer(bootstrap_servers=list(str(bootstrap_server).split(",")),
                                 value_serializer=lambda x: json.dumps(x).encode('utf-8'))
        logger.info('get_producer : producer returned successfully')
        return producer
    except Exception as e:
        logger.error('get_producer : ERROR OCCURRED while creating producer, ERROR =  ' + str(e))
        return None