import json
from kafka import KafkaProducer
from onmt.utils.logging import init_logger,logger
import os

kafka_ip_host = 'kafka_ip_host'
default_value = 'localhost:9092'
bootstrap_server = os.environ.get(kafka_ip_host,default_value)


def get_producer():
    try:
        producer = KafkaProducer(bootstrap_servers=[bootstrap_server],
                                 value_serializer=lambda x: json.dumps(x).encode('utf-8'))
        logger.info('get_producer : producer returned successfully')
        return producer
    except Exception as e:
        logger.error('get_producer : ERROR OCCURRED while creating producer, ERROR =  ' + str(e))
        return None