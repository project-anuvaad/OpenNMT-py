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

kafka_topic = [
  {
    "consumer":"to-nmt",
    "producer":"listener",
    "description":"Document translation,also used in Suvas"
  },
  {
    "consumer":"nmt_translate",
    "producer":"nmt_translate_processed",
    "description":"Pdf dev environment translation"
  },
  {
    "consumer":"nmt_translate_production",
    "producer":"nmt_translate_processed_production",
    "description":"Pdf production translation"
  },
  {
    "consumer":"anuvaad_nmt_translate",
    "producer":"anuvaad_nmt_translate_processed",
    "description":"kafka topics for"
  }

  ]
