
from flask import Flask, render_template, request, jsonify
from flask import Flask
import translation_util.interactive_translate as interactive_translation
from flask_cors import CORS
from onmt.utils.logging import init_logger
from config.config import statusCode

INTERACTIVE_LOG_FILE = 'available_models/models/interactive_log_file.txt'

logger = init_logger(INTERACTIVE_LOG_FILE)

app = Flask(__name__)
CORS(app)

@app.route('/interactive-translation', methods=['POST'])
def translate():
    inputs = request.get_json(force=True)
    if len(inputs)>0:
        logger.info("Making interactive-translation API call")
        logger.info("inputs---{}".format(inputs))
        out = interactive_translation.interactive_translation(inputs)
        complete_response = out['response_body']
        out['response_body'] = [{"tgt": complete_response[i]['tgt'][0],"tagged_tgt":complete_response[i]['tagged_tgt'][0],
                                "tagged_src":complete_response[i]['tagged_src'],"s_id":complete_response[i]['s_id'],
                                "src":complete_response[i]["src"]}
                for i in range(len(complete_response))]
        logger.info("out from interactive-translation done{}".format(out))
        return jsonify(out)
    else:
        logger.info("null inputs in request in interactive-translation API")
        return jsonify({'status':statusCode["INVALID_API_REQUEST"]})  

@app.route('/interactive-model-convert', methods=['POST'])
def model_converter():
    inputs = request.get_json(force=True)
    if len(inputs)>0:
        logger.info("Making interactive-model-convert API call")
        out = interactive_translation.model_conversion(inputs)
        return jsonify(out)
    else:
        logger.info("null inputs in request in interactive-translation API")
        return jsonify({'status':statusCode["INVALID_API_REQUEST"]}) 

@app.route('/v1/interactive-translation', methods=['POST'])
def multiple_translate():
    inputs = request.get_json(force=True)
    if len(inputs)>0:
        logger.info("Making v1/interactive-translation API call")
        logger.info("inputs---{}".format(inputs))
        out = interactive_translation.interactive_translation(inputs)
        logger.info("out from v1/interactive-translation done{}".format(out))
        return jsonify(out)
    else:
        logger.info("null inputs in request in v1/interactive-translation API")
        return jsonify({'status':statusCode["INVALID_API_REQUEST"]})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
    