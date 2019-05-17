statusCode = {
  "SUCCESS": { "ok": "true",  "statusCode": 200 , "why": "Requested operation successful" },
  "FILE_MISSING": { "ok": "false", "statusCode": 404 , "why": "No file found" },
  "TYPE_MISSING": { "ok": "false", "statusCode": 404 , "why": "No type found in the request" },
  "INVALID_TYPE": { "ok": "false",  "statusCode": 401 , "why": "Invalid file type of file to be downloaded/uploaded !","errObj":{} },
  "SYSTEM_ERR": { "ok": "false",  "statusCode": 500 , "why": "Something went wrong on the server !","errObj":{} },
  "SEVER_MODEL_ERR": { "ok": "false",  "statusCode": 500 , "why": "Something went wrong on the server !","errObj":{} },

}
