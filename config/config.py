statusCode = {
  "SUCCESS": { "ok": "true",  "statusCode": 200 , "why": "Requested operation successful" },
  "FILE_MISSING": { "ok": "false", "statusCode": 404 , "why": "No file found" },
  "TYPE_MISSING": { "ok": "false", "statusCode": 404 , "why": "No type found in the request" },
  "ID_MISSING": { "ok": "false", "statusCode": 404 , "why": "No ID found in the request" },
  "LANGUAGE_MISSING": { "ok": "false", "statusCode": 404 , "why": "No language found in the request" },
  "MANDATORY_PARAM_MISSING": { "ok": "false", "statusCode": 404 , "why": "missing mandatory params from type','language','id" },
  "TYPE_OR_LANGUAGE_MISSING": { "ok": "false", "statusCode": 404 , "why": "either type or language missing in form data" },
  "INVALID_TYPE": { "ok": "false",  "statusCode": 401 , "why": "Invalid file type of file to be downloaded/uploaded !" },
  "SYSTEM_ERR": { "ok": "false",  "statusCode": 500 , "why": "Something went wrong on the server !" },
  "SEVER_MODEL_ERR": { "ok": "false",  "statusCode": 500 , "why": "Something went wrong on the server !" },
  "UNSUPPORTED_LANGUAGE": { "ok": "false",  "statusCode": 401 , "why": "only hindi and english languages are supported" },
  "No_File_DB": { "ok": "false",  "statusCode": 401 , "why": "no file found in the db for the given id" },
  "ID_OR_SRC_MISSING": { "ok": "false",  "statusCode": 401 , "why": "Either id or src missing for some inputs in the request" },
  "INCORRECT_ID": { "ok": "false",  "statusCode": 401 , "why": "wrong model id for some input" },
  "INVALID_API_REQUEST": {"ok": "false",  "statusCode": 401 , "why": "invalid api request,either incorrect format or empty request"},
  "KAFKA_INVALID_REQUEST": {"ok": "false",  "statusCode": 401 , "why": "incorrect url_end_point for KAFKA"}

}

benchmark_types = ['Gen','LC','GoI','TB']

language_supported = ['hindi','english','tamil']

file_location ={
  "HINDI_BENCHMARK":"benchmark/hindi/",
  "ENGLISH_BENCHMARK":"benchmark/english/",
  "FILE_LOC":"benchmark/"
}

model_id = {
  "HINDI_TO_ENG":[2,3,4],
  "ENG_TO_HIN":[1]
}

## 3 is the oldest, 2 is sp ,4 is subword