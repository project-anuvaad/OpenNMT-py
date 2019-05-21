#!/usr/bin/env python
from __future__ import unicode_literals
import configargparse
import sys
from config.config import statusCode
import bleu_results as bleu_results

from flask import Flask, jsonify, request,send_file,abort,send_from_directory
from flask_cors import CORS
from onmt.translate import TranslationServer, ServerModelError

from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator
import os
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser

STATUS_OK = "ok"
STATUS_ERROR = "error"

API_FILE_DIRECTORY = "src_tgt_api_files/"

if not os.path.exists(API_FILE_DIRECTORY):
    os.makedirs(os.path.join(API_FILE_DIRECTORY,'source_files/'))
    os.makedirs(os.path.join(API_FILE_DIRECTORY,'target_files/'))
    os.makedirs(os.path.join(API_FILE_DIRECTORY,'target_ref_files/'))

def start(config_file,
          url_root="/translator",
          host="0.0.0.0",
          port=6000,
          debug=True):
    def prefix_route(route_function, prefix='', mask='{0}{1}'):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)
        return newroute

    app = Flask(__name__)
    CORS(app)
    app.route = prefix_route(app.route, url_root)
    translation_server = TranslationServer()
   ## translation_server.start(config_file)##

    @app.route('/models', methods=['GET'])
    def get_models():
        out = translation_server.list_models()
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

    @app.route('/translate', methods=['POST'])
    def translate():
        print("here1")
        inputs = request.get_json(force=True)
        print(inputs)
        out = {}
        try:
            translation, scores, n_best, times = translation_server.run(inputs)
            assert len(translation) == len(inputs)
            assert len(scores) == len(inputs)

            out = [[{"src": inputs[i]['src'], "tgt": translation[i],
                     "n_best": n_best,
                     "pred_score": scores[i]}
                    for i in range(len(translation))]]
        except ServerModelError as e:
            out['error'] = str(e)
            out['status'] = STATUS_ERROR

        return jsonify(out)
    @app.route('/translation', methods=['POST'])
    def translation():
        print("custom made function")
        inputs = request.get_json(force=True)
        print(inputs)
        out = {}
        try:
            with open("~/test/apiInput.txt", "w") as text_file:
                text_file.write(str(inputs[0]['src']))
            os.system('python ~/indic_nlp_library/src/indicnlp/tokenize/indic_tokenize.py ~/test/apiInput.txt ~/test/apiInputTok.txt hi')
            os.system('./tools/apply_bpe.py -c ~./tools/codesSrc0605.bpe < ~/test/apiInputTok.txt > ~/test/apiInputTokBpe0605.txt')
            os.system('python translate.py -model model/model_100105-model_step_90000.pt -src ~/test/apiInputTokBpe0605.txt -output mypredifTok.txt -replace_unk -verbose')
            os.system("sed -r 's/(@@ )|(@@ ?$)//g' mypredifTok.txt > finaltranslationEndeBpe90k0605.txt")
            os.system("perl ./tools/detokenize.perl <finaltranslationEndeBpe90k0605.txt> finaltranslationEndeBpedeTok90k0605.txt -l en")
            os.system("perl ./tools/detrucaser.perl  <finaltranslationEndeBpedeTok90k0605.txt>finaltranslationEndeBpedeTokdeTru90k0605.txt")
            with open("finaltranslationEndeBpedeTokdeTru90k0605.txt") as zh:
                out = zh.readlines()
            #return send_file('/home/ubuntu/OpenNMT-py/mypredif.txt')
                return jsonify(out)
        except ServerModelError as e:
            out = statusCode["SEVER_MODEL_ERR"]
            out['errObj'] = str(e)
            # out['status'] = STATUS_ERROR
            return jsonify(out)
        except:
            out = statusCode["SYSTEM_ERR"]
            print("Unexpected error:", sys.exc_info()[0])
            return jsonify(out)     

    @app.route("/download-src", methods=['GET'])
    def get_file():
        """Download a file."""
        out = {}
        type = request.args.get('type')
        print(type)
        if  not type:
            out = statusCode["TYPE_MISSING"]
            return jsonify(out)
        if type not in ['Gen','LC','GoI','TB']:
            out = statusCode["INVALID_TYPE"]
            return jsonify(out)  

        try:
            print("downloading the src %s.txt file" % type)
            return send_file(os.path.join(API_FILE_DIRECTORY,'source_files/', '%s.txt' % type), as_attachment=True)
        except:
            out = statusCode["SYSTEM_ERR"]
            print("Unexpected error:", sys.exc_info()[0])
            return jsonify(out) 

    @app.route("/upload-tgt", methods=["POST"])
    def post_file():
        """Upload a file."""
        print(request.files)
        out = {}
        if 'file' not in request.files:
            out = statusCode["FILE_MISSING"]
            return jsonify(out)
        print(request.form)    
        if 'type' not in request.form:
            out = statusCode["TYPE_MISSING"]
            return jsonify(out)  
        if request.form['type'] not in ['Gen','LC','GoI','TB']:
            out = statusCode["INVALID_TYPE"]
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
            print("Bleu calculated and file removed")
            with open("bleu-detok.txt") as zh:
                out = statusCode["SUCCESS"]
                # out['bleu_for_uploaded_file'] = float(', '.join(zh.readlines()))
                out['bleu_for_uploaded_file'] = float(', '.join(zh.readlines()))
                out['openNMT_custom'] = bleu_results.OpenNMT_Custom
                out['google_api'] = bleu_results.GOOGLE_API
                return jsonify(out)
        except:
            out = statusCode["SYSTEM_ERR"]
            print("Unexpected error:", sys.exc_info()[0])
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
    parser.add_argument("--port", type=int, default="6000")
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
