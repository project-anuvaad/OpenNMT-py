#!/usr/bin/env python
from __future__ import unicode_literals
import configargparse
import sys
from config.config import statusCode,benchmark_types, language_supported, file_location
import bleu_results as bleu_results
import anuvada
import tools.sp_enc_dec as sp
import ancillary_functions_anuvaad.ancillary_functions as ancillary_functions
import ancillary_functions_anuvaad.sc_preface_handler as sc_preface_handler
import ancillary_functions_anuvaad.handle_date_url as date_url_util

from flask import Flask, jsonify, request,send_file,abort,send_from_directory
from flask_cors import CORS
from onmt.translate import TranslationServer, ServerModelError

from itertools import repeat

from onmt.utils.logging import init_logger,logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
import os
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from mongo_model import db,Benchmarks
import datetime
from kafka_utils.document_translator import doc_translator
import threading
import translation_util.translate_util as translate_util

STATUS_OK = "ok"
STATUS_ERROR = "error"

API_FILE_DIRECTORY = "src_tgt_api_files/"
mongo_config_dir = "config/mongo_config.py"
IS_RUN_KAFKA = 'IS_RUN_KAFKA'
IS_RUN_KAFKA_DEFAULT_VALUE = False
bootstrap_server_boolean = os.environ.get(IS_RUN_KAFKA, IS_RUN_KAFKA_DEFAULT_VALUE)

if not os.path.exists(API_FILE_DIRECTORY):
    os.makedirs(os.path.join(API_FILE_DIRECTORY,'source_files/'))
    os.makedirs(os.path.join(API_FILE_DIRECTORY,'target_files/'))
    os.makedirs(os.path.join(API_FILE_DIRECTORY,'target_ref_files/'))

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
        logger.info('starting kafka from nmt-server')
        doc_translator(translation_server) 

    if bootstrap_server_boolean:
        t1 = threading.Thread(target=kafka_function)
        t1.start()

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
            out = translate_util.translate_func(inputs, translation_server)
            logger.info("out from translate_func-trans_util done{}".format(out))
            return jsonify(out)
        else:
            logger.info("null inputs in request in translate-anuvaad API")
            return jsonify({'status':statusCode["INVALID_API_REQUEST"]})       

    @app.route('/translation_en', methods=['POST'])
    def translation_en():
        inputs = request.get_json(force=True)
        if len(inputs)>0:
            logger.info("Making translation_en API call")
            out = translate_util.from_en(inputs, translation_server)
            logger.info("out from english-trans_util done{}".format(out))
            return jsonify(out)  
        else:
            logger.info("null inputs in request in translation_en API")
            return jsonify({'status':statusCode["INVALID_API_REQUEST"]})       

    @app.route('/translation_hi', methods=['POST'])
    def translation_hi():
        inputs = request.get_json(force=True)
        if len(inputs)>0:
            logger.info("Making translation_hi API call")
            out = translate_util.from_hindi(inputs, translation_server)
            logger.info("out from hindi-trans_util done{}".format(out))
            return jsonify(out)
        else:
            logger.info("null inputs in request in translation_hi API")
            return jsonify({'status':statusCode["INVALID_API_REQUEST"]})  

    @app.route('/save_benchmark', methods=['POST'])
    def save_benchmark():
        out = {}
        # print(inputs = request.get_json(force=True))
        if 'file' not in request.files:
            out['status'] = statusCode["FILE_MISSING"]
            return jsonify(out)
            
        if  any(v not in request.form for v in ['type','language']):
            out['status'] = statusCode["TYPE_OR_LANGUAGE_MISSING"]
            return jsonify(out) 
        if request.form['type'] not in benchmark_types:
            out['status'] = statusCode["INVALID_TYPE"]
            return jsonify(out)
        if request.form['language'] not in language_supported:
            out['status'] = statusCode["UNSUPPORTED_LANGUAGE"]
            return jsonify(out)    
        try:
            file_type = request.form['type']
            file = request.files['file']
            language = request.form['language']
            user_filename = file.filename
            db_filename = user_filename +"-"+ str(datetime.datetime.now().timestamp())

            if not os.path.exists(os.path.join(file_location['FILE_LOC'],'%s/'%language)):
                os.makedirs(os.path.join(file_location['FILE_LOC'],'%s/'%language)) 
            file_loc =  os.path.join(file_location["FILE_LOC"], language,db_filename)
            Benchmarks(type = file_type,language = request.form['language'],user_filename = user_filename,db_filename = db_filename,version = 0 ,created_by = "", path = file_loc).save()

            file.save(file_loc)
            
            logger.info("saving file name:%s ,type:%s"%(user_filename,file_type))
            out['status'] = statusCode["SUCCESS"]
            out['response_body'] = {}
        except Exception as e:
            out['status'] = statusCode["SYSTEM_ERR"]
            out['status']['errObj'] = str(e)
            logger.info("Unexpected error: %s"% sys.exc_info()[0]) 
        
        return jsonify(out)

    @app.route('/list_benchmark', methods=['GET'])
    def list_benchmark():
        out = {}        
        if  'language' not in request.args:
            out['status'] = statusCode["LANGUAGE_MISSING"]
            return jsonify(out)
        if request.args.get('language') not in language_supported:
            out['status'] = statusCode["UNSUPPORTED_LANGUAGE"]
            return jsonify(out) 
        try:  
            language = request.args.get('language')         
            list_benchmark = Benchmarks.objects(language = language).exclude('path').exclude('db_filename')   
            
            logger.info("listing benchmark files for %s language" % language)
            out['status'] = statusCode["SUCCESS"]
            out['response_body'] = {"list_benchmark":list_benchmark}
        except Exception as e:
            out['status'] = statusCode["SYSTEM_ERR"]
            out['status']['errObj'] = str(e)
            logger.info("Unexpected error: %s"% sys.exc_info()[0])
        
        return jsonify(out)

    @app.route("/download_benchmark", methods=['GET'])
    def download_benchmark():
        out = {}
        if  'id' not in request.args:
            out['status'] = statusCode["MANDATORY_PARAM_MISSING"]
            return jsonify(out)

        try:
            id = request.args.get('id')
            path =  Benchmarks.objects(id =id).only('path')
            
            if len(path) > 0:
                path = path[0].path
                logger.info("downloading the benchmark file %s file" % path)
                return send_file(path, as_attachment=True)
            else:
                out['status'] =  statusCode["No_File_DB"]           
            
        except Exception as e:
            out['status'] = statusCode["SYSTEM_ERR"]
            out['status']['errObj'] = str(e)
            logger.info("Unexpected error: %s"% sys.exc_info()[0])
        return jsonify(out)
    
    @app.route("/calculate_bleu", methods=["POST"])
    def calculate_bleu():     
        out = {}
       
        if 'file' not in request.files:
            out['status'] = statusCode["FILE_MISSING"]
            return jsonify(out)
        if  'id' not in request.form:
            out['status'] = statusCode["MANDATORY_PARAM_MISSING"]
            return jsonify(out)
        
        try:
            id = request.form['id']
            file = request.files['file']
            tgt_file_loc = os.path.join('intermediate_data', '%s.txt' % file.filename)
            file.save(tgt_file_loc)
            path =  Benchmarks.objects(id =id).only('path')
            if len(path) > 0:
                path = path[0].path
                logger.info("calculating bleu using %s file" % path) 
                bleu_file = os.popen("perl ./tools/multi-bleu-detok.perl %s < %s " %(path,tgt_file_loc)).read()    
                os.remove(tgt_file_loc)
                logger.info("Bleu calculated and file removed")
                out['status'] = statusCode["SUCCESS"]
                out['response_body'] = {'bleu_for_uploaded_file':float(bleu_file),
                                        'openNMT_custom':bleu_results.OpenNMT_Custom, 'google_api': bleu_results.GOOGLE_API 
                                        }                
            else:
                out['status'] =  statusCode["No_File_DB"]   

        except Exception as e:
            out['status'] = statusCode["SYSTEM_ERR"]
            out['status']['errObj'] = str(e)
            logger.info("Unexpected error: %s"% sys.exc_info()[0])
        return jsonify(out)

    @app.route("/download-src", methods=['GET'])
    def get_file():
        """Download a file."""
        out = {}
        type = request.args.get('type')
        print(type)
        if  not type:
            out['status'] = statusCode["TYPE_MISSING"]
            return jsonify(out)
        if type not in ['Gen','LC','GoI','TB']:
            out['status'] = statusCode["INVALID_TYPE"]
            return jsonify(out)  

        try:
            logger.info("downloading the src %s.txt file" % type)
            return send_file(os.path.join(API_FILE_DIRECTORY,'source_files/', '%s.txt' % type), as_attachment=True)
        except:
            out['status'] = statusCode["SYSTEM_ERR"]
            logger.info("Unexpected error: %s"% sys.exc_info()[0])
            return jsonify(out) 

    @app.route("/upload-tgt", methods=["POST"])
    def post_file():
        """Upload a file."""
        print(request.files)
        out = {}
        if 'file' not in request.files:
            out['status'] = statusCode["FILE_MISSING"]
            return jsonify(out)
        print(request.form)    
        if 'type' not in request.form:
            out['status'] = statusCode["TYPE_MISSING"]
            return jsonify(out)  
        if request.form['type'] not in ['Gen','LC','GoI','TB']:
            out['status'] = statusCode["INVALID_TYPE"]
            return jsonify(out)  

        try:
            file = request.files['file']
            tgt_file_loc = os.path.join(API_FILE_DIRECTORY,'target_files/', '%s.txt' % request.form['type'])
            tgt_ref_file_loc = os.path.join(API_FILE_DIRECTORY,'target_ref_files/', '%s.txt' % request.form['type'])
            file.save(tgt_file_loc)

            if os.path.exists("bleu_out.txt"):
               os.remove("bleu_out.txt")
            
            os.system("perl ./tools/multi-bleu-detok.perl ./%s < ./%s > bleu-detok.txt" %(tgt_ref_file_loc,tgt_file_loc))
            os.system("python ./tools/calculatebleu.py ./%s ./%s" %(tgt_file_loc,tgt_ref_file_loc))            
            os.remove(tgt_file_loc)
            logger.info("Bleu calculated and file removed")
            with open("bleu-detok.txt") as zh:
                out['status'] = statusCode["SUCCESS"]
                out['response_body'] = {'bleu_for_uploaded_file':float(', '.join(zh.readlines())),
                                        'openNMT_custom':bleu_results.OpenNMT_Custom, 'google_api': bleu_results.GOOGLE_API 
                                        }
        except:
            out['status'] = statusCode["SYSTEM_ERR"]
            logger.info("Unexpected error: %s"% sys.exc_info()[0])
        
        return jsonify(out)    

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
