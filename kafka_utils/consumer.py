from kafka import KafkaConsumer
from onmt.utils.logging import init_logger,logger
import json
import os

KAFKA_IP_HOST = 'KAFKA_IP_HOST'
default_value = 'localhost:9092'
bootstrap_server = os.environ.get(KAFKA_IP_HOST, default_value)


def get_consumer(topic):
    try:
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[bootstrap_server],
            auto_offset_reset='earliest',
            value_deserializer=lambda x: json.loads(x.decode('utf-8')))

        logger.info('get_consumer : consumer returned for topic = ' + topic)
        return consumer
    except Exception as e:
        logger.error(
            'get_consumer : ERROR OCCURRED for getting consumer with topic = ' + topic)
        logger.error('get_consumer : ERROR = ' + str(e))
        print('error')
        return None
