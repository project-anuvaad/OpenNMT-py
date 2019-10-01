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

STATUS_OK = "ok"
STATUS_ERROR = "error"

API_FILE_DIRECTORY = "src_tgt_api_files/"
mongo_config_dir = "config/mongo_config.py"

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
        logger.info('kafka_function, in server')
        doc_translator(translation_server)

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

    @app.route('/translate', methods=['POST'])
    def translate():
        ## not using
        inputs = request.get_json(force=True)
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

    @app.route('/translation_en', methods=['POST'])
    def translation_en():
        inputs = request.get_json(force=True)
        out = {}
        tgt = list()
        pred_score = list()
        sentence_id = list()
        input_subwords = list()
        output_subwords = list()
        try:
            for i in inputs:
                if 's_id'in i:
                    s_id = [i['s_id']]
                else:
                    s_id = [0000]    
                if  any(v not in i for v in ['src','id']):
                    out['status'] = statusCode["ID_OR_SRC_MISSING"]
                    return jsonify(out)

                # if len(i['src'].split()) == 1 and i['src'].isalpha()== False:
                #     logger.info("handling single token")
                #     translation = ancillary_functions.handle_single_token(i['src'])
                #     scores = [1]
                # if len(i['src'].split()) == 1 and i['src'].isalpha() and len(i['src'])==1:
                #     logger.info("returning single character as it is:%s"%i['src'])
                #     translation = i['src']
                #     scores = [1]
                i['src'] = i['src'].strip()
                if ancillary_functions.special_case_fits(i['src']):
                    logger.info("sentence fits in special case, returning accordingly and not going to model")
                    translation = ancillary_functions.handle_special_cases(i['src'],i['id'])
                    scores = [1]      
                else:
                    logger.info("translating using NMT-models")
                    # prefix,suffix, i['src'] = ancillary_functions.separate_alphanumeric_and_symbol(i['src'])
                    # print("prefix :{},suffix :{},i[src] :{}".format(prefix,suffix,i['src']))
                    # i['src'] = anuvada.moses_tokenizer(i['src'])
                    # i['src'] = anuvada.truecaser(i['src'])  
                    if i['id'] == 1:
                        i['src'] = anuvada.moses_tokenizer(i['src'])                   
                        i['src'] = str(sp.encode_line('en-220519.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('hi-220519.model',translation[0])
                        translation = anuvada.indic_detokenizer(translation)
                    elif i['id'] == 9:   
                        i['src'] = anuvada.moses_tokenizer(i['src'])                
                        i['src'] = str(sp.encode_line('enT-08072019-10k.model',i['src']))
                        translation, scores, n_best, times = translation_server.run([i])
                        translation = sp.decode_line('ta-08072019-10k.model',translation[0])
                    elif i['id'] == 8:
                        numbers = sc_preface_handler.get_numbers(i['src'])
                        i['src'] = sc_preface_handler.replace_numbers_with_hash(i['src'])
                        i['src'] = str(sp.encode_line('enSC-02082019-2k.model',i['src']))
                        translation, scores, n_best, times = translation_server.run([i])
                        translation = sp.decode_line('hiSC-02082019-1k.model',translation[0])
                        translation = sc_preface_handler.replace_hash_with_original_number(translation,numbers)  
                    # elif i['id'] == 7:  
                    #     i['src'],date_original,url_original = date_url_util.tag_number_date_url(i['src'])   
                    #     print("herere")           
                    #     i['src'] = str(sp.encode_line('model/sentencepiece_models/enT-210819-7k.model',i['src']))
                    #     input_sw = i['src']
                    #     translation, scores, n_best, times = translation_server.run([i])
                    #     output_sw = translation[0]
                    #     translation = sp.decode_line('model/sentencepiece_models/ta-210819-7k.model',translation[0])
                    #     translation = date_url_util.replace_tags_with_original(translation,date_original,url_original)
                    elif i['id'] == 10:  
                        "english-gujrati"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])          
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/en-2019-09-10-10k.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/guj-2019-09-10-10k.model',translation[0])
                        logger.info("decoded gujrati: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 11:  
                        "english-bengali"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])           
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/en-2019-09-12-10k.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/beng-2019-09-12-10k.model',translation[0])
                        logger.info("decoded bengali: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array) 
                    elif i['id'] == 12:  
                        "english-marathi"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])             
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/enMr-2019-09-14-10k.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/marathi-2019-09-14-10k.model',translation[0])
                        logger.info("decoded marathi: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] in [13,14]:  
                        "170919 eng-hin"
                        i['src'] = i['src'].lower()
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])            
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/en-2019-09-17-10k.model',i['src']))
                        input_sw = i['src']
                        logger.info("encoded english: {}".format( i['src']))
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/hi-2019-09-17-10k.model',translation[0])
                        logger.info("decoded hindi: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)                   

                    elif i['id'] == 15:  
                        "english-kannada"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])             
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/enKn-2019-09-20-10k.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/kannada-2019-09-20-10k.model',translation[0])
                        logger.info("decoded kannada: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 16:  
                        "english-telgu"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])             
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/enTe-2019-09-20-10k.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/telgu-2019-09-20-10k.model',translation[0])
                        logger.info("decoded telgu: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 17:  
                        "english-malayalam"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])             
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/enMl-2019-09-20-10k.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/malayalam-2019-09-20-10k.model',translation[0])
                        logger.info("decoded malayalam: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 18:  
                        "english-punjabi"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])             
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/enPu-2019-09-20-10k.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/punjabi-2019-09-20-10k.model',translation[0])
                        logger.info("decoded punjabi: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 7:  
                        "english-tamil"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])             
                        i['src'] = str(sp.encode_line('model/sentencepiece_models/enTa-2019-09-23-10k.model',i['src']))
                        input_sw = i['src']
                        translation, scores, n_best, times = translation_server.run([i])
                        output_sw = translation[0]
                        translation = sp.decode_line('model/sentencepiece_models/tamil-2019-09-23-10k.model',translation[0])
                        logger.info("decoded tamil: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)    
                    else:
                        out['status'] = statusCode["INCORRECT_ID"]
                        return jsonify(out)
                    
                    # translation = (prefix+" "+translation+" "+suffix).strip()
                translation = ancillary_functions.replace_hindi_numbers(translation)
                tgt.append(translation)
                pred_score.append(scores[0])
                sentence_id.append(s_id[0])
                input_subwords.append(input_sw)
                output_subwords.append(output_sw)

            out['status'] = statusCode["SUCCESS"]
            out['response_body'] = [{"tgt": tgt[i],
                                     "s_id": sentence_id[i],"input_subwords": input_subwords[i], 
                                     "output_subwords":output_subwords[i]}
                    for i in range(len(tgt))]
        except ServerModelError as e:
            out['status'] = statusCode["SEVER_MODEL_ERR"]
            out['status']['errObj'] = str(e)
        except Exception as e:
            out['status'] = statusCode["SYSTEM_ERR"]
            print("error: {}".format(e))
            logger.info("Unexpected error: %s"% sys.exc_info()[0])    

        return jsonify(out)        

    @app.route('/translation_hi', methods=['POST'])
    def translation_hi():
        inputs = request.get_json(force=True)
        out = {}
        tgt = list()
        pred_score = list()
        sentence_id = list()
        try:
            for i in inputs:
                if 's_id'in i:
                    s_id = [i['s_id']]
                else:
                    s_id = [0000]   
                if  any(v not in i for v in ['src','id']):
                    out['status'] = statusCode["ID_OR_SRC_MISSING"]
                    return jsonify(out) 
                if i['id'] == 3:
                    logger.info("translating using the first model")
                    translation, scores, n_best, times = translation_server.run([i])
                    translation = translation[0]   

                else:
                    i['src'] = anuvada.indic_tokenizer(i['src']) 

                    if i['id'] == 2:
                        i['src'] = str(sp.encode_line('hi-220519.model',i['src']))
                        translation, scores, n_best, times = translation_server.run([i])
                        translation = sp.decode_line('en-220519.model',translation[0])
                    elif i['id'] in [5,6]:
                        i['src'] = str(sp.encode_line('hi-28062019-10k.model',i['src']))
                        translation, scores, n_best, times = translation_server.run([i])
                        translation = sp.decode_line('en-28062019-10k.model',translation[0])
                    elif i['id'] == 4:
                        i['src'] = anuvada.apply_bpe('codesSrc1005.bpe',i['src'])
                        translation, scores, n_best, times = translation_server.run([i])
                        translation = anuvada.decode_bpe(translation[0])
                    else:
                        out['status'] = statusCode["INCORRECT_ID"]
                        return jsonify(out)      
                    translation = anuvada.moses_detokenizer(translation)
                    translation = anuvada.detruecaser(translation)
                
                tgt.append(translation)
                pred_score.append(scores[0])
                sentence_id.append(s_id[0])

            out['status'] = statusCode["SUCCESS"]
            out['response_body'] = [{"tgt": tgt[i],
                     "pred_score": pred_score[i], "s_id": sentence_id[i]}
                    for i in range(len(tgt))]
        except ServerModelError as e:
            out['status'] = statusCode["SEVER_MODEL_ERR"]
            out['status']['errObj'] = str(e)
        except Exception as e:
            out['status'] = statusCode["SYSTEM_ERR"]
            out['status']['errObj'] = str(e)
            logger.info("Unexpected error: %s"% sys.exc_info()[0])   

        return jsonify(out)

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
