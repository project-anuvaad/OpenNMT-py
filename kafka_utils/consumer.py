from kafka import KafkaConsumer
from onmt.utils.logging import init_logger,logger
import json
import os

KAFKA_IP_HOST = 'KAFKA_IP_HOST'
default_value = 'localhost:9092'
bootstrap_server = os.environ.get(KAFKA_IP_HOST, default_value)
group_id = 'anuvaad'


def get_consumer(topics):
    try:
        # consumer = KafkaConsumer(
        #            topic,
        #            bootstrap_servers=[bootstrap_server],
        #            auto_offset_reset='earliest',
        #            enable_auto_commit=True,
        #            group_id=group_id,
        #            value_deserializer=lambda x: json.loads(x.decode('utf-8')))
        consumer = KafkaConsumer(
            bootstrap_servers=list(str(bootstrap_server).split(",")),
            value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    
        consumer.subscribe(topics)    
        logger.info('get_consumer : consumer returned for topics:{}'.format(topics))
        return consumer
    except Exception as e:
        logger.error('ERROR OCCURRED for getting consumer with topics:{}'.format(topics))
        logger.error('get_consumer : ERROR = ' + str(e))
        return None