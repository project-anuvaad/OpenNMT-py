import os

consumer_topics ={
  "TEST_TOPIC":"testtopic",
  "DOCUMENT_REQ":"to-nmt",
  "new_topic":"nmt_translate"
}

producer_topics ={
  "TEST_TOPIC":"listener",
  "TO_DOCUMENT":"listener",
  "new_topic":"nmt_translate_processed"
}

## "description":"default topics"
nmt_input_topic_default = "anuvaad-nmt-input-default"
nmt_output_topic_default = 'anuvaad-nmt-output-default'

kafka_topic = [
  {
    "consumer":os.environ.get('KAFKA_ANUVAAD_DOC_INPUT_TOPIC', nmt_input_topic_default),
    "producer":os.environ.get('KAFKA_ANUVAAD_DOC_OUTPUT_TOPIC', nmt_output_topic_default),
    "description":"Document translation,also used in Suvas"
  },
  {
    "consumer":os.environ.get('KAFKA_ANUVAAD_PDF_INPUT_TOPIC', nmt_input_topic_default),
    "producer":os.environ.get('KAFKA_ANUVAAD_PDF_OUTPUT_TOPIC', nmt_output_topic_default),
    "description":"Pdf without WFM translation"
  },
  {
    "consumer":os.environ.get('KAFKA_ANUVAAD_WFM_INPUT_TOPIC', nmt_input_topic_default),
    "producer":os.environ.get('KAFKA_ANUVAAD_WFM_OUTPUT_TOPIC', nmt_output_topic_default),
    "description":"kafka topics with WFM"
  }
]

