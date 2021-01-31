#!/usr/bin/env python
from __future__ import unicode_literals
import configargparse
import sys
from config.config import statusCode,benchmark_types, language_supported, file_location
import config.bleu_results as bleu_results
import tools.sp_enc_dec as sp
import ancillary_functions_anuvaad.ancillary_functions as ancillary_functions
import ancillary_functions_anuvaad.sc_preface_handler as sc_preface_handler
import ancillary_functions_anuvaad.handle_date_url as date_url_util

from flask import Flask, jsonify, request,send_file,abort,send_from_directory
from flask_cors import CORS
from onmt.translate import TranslationServer, ServerModelError

from itertools import repeat

from onmt.utils.logging import init_logger,logger,entry_exit_log,LOG_TAGS
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
import os
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from config.mongo_model import db,Benchmarks
import datetime
from kafka_utils.document_translator import doc_translator
import threading
import translation_util.translate_util as translate_util
import translation_util.interactive_translate as interactive_translation
from config.kafka_topics import consumer_topics,producer_topics,kafka_topic

STATUS_OK = "ok"
STATUS_ERROR = "error"

mongo_config_dir = "config/mongo_config.py"
IS_RUN_KAFKA = 'IS_RUN_KAFKA'
IS_RUN_KAFKA_DEFAULT_VALUE = False
bootstrap_server_boolean = os.environ.get(IS_RUN_KAFKA, IS_RUN_KAFKA_DEFAULT_VALUE)


def start(config_file,
          url_root="/translator",
          host="0.0.0.0",
          port=3003,
          debug=True):
    def prefix_route(route_function, prefix='', mask='{0}{1}'):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)
        return newroute

    app = Flask(__name__)
    CORS(app)
    app.config.from_pyfile(mongo_config_dir)
    db.init_app(app)
    app.route = prefix_route(app.route, url_root)
    translation_server = TranslationServer()
    translation_server.start(config_file)

    def kafka_function():
        logger.info('starting kafka from nmt-server on thread-1')
        doc_translator(translation_server,[kafka_topic[0]['consumer'],kafka_topic[1]['consumer'],kafka_topic[2]['consumer']])     

    if bootstrap_server_boolean:
        t1 = threading.Thread(target=kafka_function)
        # t1.start()

    @app.route('/models', methods=['GET'])
    def get_models():
        out = {}
        try:
            out['status'] = statusCode["SUCCESS"]
            out['response_body'] = translation_server.list_models()
        except:
            out['status'] = statusCode["SYSTEM_ERR"]
            logger.info("Unexpected error: %s"% sys.exc_info()[0]) 
        return jsonify(out)

    @app.route('/clone_model/<int:model_id>', methods=['POST'])
    def clone_model(model_id):
        out = {}
        data = request.get_json(force=True)
        timeout = -1
        if 'timeout' in data:
            timeout = data['timeout']
            del data['timeout']

        opt = data.get('opt', None)
        try:
            model_id, load_time = translation_server.clone_model(
                model_id, opt, timeout)
        except ServerModelError as e:
            out['status'] = STATUS_ERROR
            out['error'] = str(e)
        else:
            out['status'] = STATUS_OK
            out['model_id'] = model_id
            out['load_time'] = load_time

        return jsonify(out)

    @app.route('/unload_model/<int:model_id>', methods=['GET'])
    def unload_model(model_id):
        out = {"model_id": model_id}

        try:
            translation_server.unload_model(model_id)
            out['status'] = STATUS_OK
        except Exception as e:
            out['status'] = STATUS_ERROR
            out['error'] = str(e)

        return jsonify(out)

    @app.route('/translate-anuvaad', methods=['POST'])
    def translate():
        inputs = request.get_json(force=True)
        if len(inputs)>0:
            logger.info("Making translate-anuvaad API call")
            logger.info(entry_exit_log(LOG_TAGS["input"],inputs))
            out = translate_util.translate_func(inputs, translation_server)
            logger.info("out from translate_func-trans_util done{}".format(out))
            logger.info(entry_exit_log(LOG_TAGS["output"],out))
            return jsonify(out)
        else:
            logger.info("null inputs in request in translate-anuvaad API")
            return jsonify({'status':statusCode["INVALID_API_REQUEST"]})                  

    @app.route('/to_cpu/<int:model_id>', methods=['GET'])
    def to_cpu(model_id):
        out = {'model_id': model_id}
        translation_server.models[model_id].to_cpu()

        out['status'] = STATUS_OK
        return jsonify(out)

    @app.route('/to_gpu/<int:model_id>', methods=['GET'])
    def to_gpu(model_id):
        out = {'model_id': model_id}
        translation_server.models[model_id].to_gpu()

        out['status'] = STATUS_OK
        return jsonify(out)

    app.run(debug=debug, host=host, port=port, use_reloader=False,
            threaded=True)


def _get_parser():
    parser = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        description="OpenNMT-py REST Server")
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="3003")
    parser.add_argument("--url_root", type=str, default="/translator")
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--config", "-c", type=str,
                        default="./available_models/conf.json")
    return parser


if __name__ == '__main__':
    parser = _get_parser()
    args = parser.parse_args()
    start(args.config, url_root=args.url_root, host=args.ip, port=args.port,
          debug=args.debug)
