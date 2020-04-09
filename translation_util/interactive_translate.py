import ctranslate2
import anuvada
import tools.sp_enc_dec as sp
import ancillary_functions_anuvaad.ancillary_functions as ancillary_functions
import ancillary_functions_anuvaad.handle_date_url as date_url_util
from config import sentencepiece_model_loc as sp_model
from config.config import statusCode
from config.regex_patterns import patterns
from onmt.utils.logging import init_logger
import json 
import sys
import os
import re

ICONFG_FILE = "available_models/interactive_models/iconf.json"
INTERACTIVE_LOG_FILE = 'intermediate_data/interactive_log_file.txt'

logger = init_logger(INTERACTIVE_LOG_FILE)


def model_conversion(inputs):
    out = {}

    if any(v not in inputs for v in ['inp_model_path','out_dir']):
        out['status'] = statusCode["INVALID_API_REQUEST"]
        logger.info("Missing either inp_model_path,out_dir in model conversion request")
        return (out)
    with open(ICONFG_FILE) as f:
        confs = json.load(f)
        model_root = confs['models_root']
    final_dir =  os.path.join(model_root, inputs['out_dir'])  
    try:
        logger.info("Inside model_conversion-interactive_translate function")
        converter = ctranslate2.converters.OpenNMTPyConverter(inputs['inp_model_path'])       # inp_model_path: the model which has to be converted
        output_dir = converter.convert(
                     final_dir,                                          # Path to the output directory.
                     "TransformerBase",                                  # A model specification instance from ctranslate2.specs.
                     vmap=None,                                          # Path to a vocabulary mapping file.
                     quantization=None,                                  # Weights quantization: "int8" or "int16".
                     force=False)
        logger.info("Interactive model converted and saved at: {}".format(output_dir))
        out['status'] = statusCode["SUCCESS"]              
    except Exception as e:
        logger.error("Error in model_conversion interactive translate: {} and {}".format(e,sys.exc_info()[0]))
        out['status'] = statusCode["SYSTEM_ERR"]
        out['status']['errObj'] = str(e)

    return (out)    


def encode_itranslate_decode(i,sp_encoder,sp_decoder,num_map):
    try:
        logger.info("Inside encode_itranslate_decode function")
        model_path = get_model_path(i['id'])
        translator = ctranslate2.Translator(model_path)
        i['src'] = str(sp.encode_line(sp_encoder,i['src']))
        i_final = format_converter(i['src'])

        if 'target_prefix' in i and len(i['target_prefix']) > 0 and i['target_prefix'].isspace() == False:
            logger.info("target prefix: {}".format(i['target_prefix'])) 
            i['target_prefix'] = i['target_prefix'].strip() 
            i['target_prefix'] = replace_num_target_prefix(i,num_map)
            i['target_prefix'] = anuvada.indic_tokenizer(i['target_prefix'])
            i['target_prefix'] = str(sp.encode_line(sp_decoder,i['target_prefix']))
            tp_final = format_converter(i['target_prefix'])
            tp_final[-1] = tp_final[-1].replace(']',",")
            m_out = translator.translate_batch([i_final],beam_size = 5, target_prefix = [tp_final])
        else:
            m_out = translator.translate_batch([i_final],beam_size = 5)

        m_tok = m_out[0][0]['tokens']  
        translation = " ".join(m_tok)
        translation = sp.decode_line(sp_decoder,translation)
        
        return translation
        
    except Exception as e:
        logger.error("Unexpexcted error in encode_itranslate_decode: {} and {}".format(e,sys.exc_info()[0]))
        raise

def interactive_translation(inputs):
    out = {}
    tgt = list()
    tagged_tgt = list()
    tagged_src = list()

    try:
        for i in inputs:            
            if  any(v not in i for v in ['src','id']):
                out['status'] = statusCode["ID_OR_SRC_MISSING"]
                logger.info("either id or src missing in some input")
                return (out) 

            logger.info("input sentences:{}".format(i['src']))    
            i['src'] = i['src'].strip()    
            if ancillary_functions.special_case_fits(i['src']):
                logger.info("sentence fits in special case, returning accordingly and not going to model")
                translation = ancillary_functions.handle_special_cases(i['src'],i['id'])

            else:
                logger.info("Performing interactive translation on:{}".format(i['id']))
                i['src'],date_original,url_original,num_array,num_map = date_url_util.tag_number_date_url_1(i['src'])
                tag_src = i['src'] 

                if i['id'] == 56:
                    i['src'] = anuvada.moses_tokenizer(i['src'])
                    translation = encode_itranslate_decode(i,sp_model.english_hindi["ENG_EXP_5.6"],sp_model.english_hindi["HIN_EXP_5.6"],num_map)
                    translation = anuvada.indic_detokenizer(translation)
                                                                   
                else:
                    logger.info("unsupported model id: {} for given input".format(i['id']))
                    raise Exception("unsupported model id: {} for given input".format(i['id']))      

                translation = date_url_util.regex_pass(translation,[patterns['p4']['regex'],patterns['p5']['regex'],patterns['p6']['regex'],patterns['p7']['regex']])
                tag_tgt = translation
                translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                logger.info("len tag_tgt-{} and len translation-{}".format(len(tag_tgt.split()),len(translation.split())))
            logger.info("interactive translation-experiment-{} output: {}".format(i['id'],translation))    
            tgt.append(translation)
            tagged_tgt.append(tag_tgt)
            tagged_src.append(tag_src)

        out['status'] = statusCode["SUCCESS"]
        out['response_body'] = [{"tgt": tgt[i],"tagged_tgt":tagged_tgt[i],
                                "tagged_src":tagged_src[i]}
                for i in range(len(tgt))]
    except Exception as e:
        out['status'] = statusCode["SYSTEM_ERR"]
        out['status']['errObj'] = str(e)
        logger.error("Unexpected error:%s and %s"% (e,sys.exc_info()[0]))   

    return (out)

def format_converter(input):
    inp_1 = input.split(', ')
    inp_2 = [inpt+',' if inpt != inp_1[-1] else inpt for inpt in inp_1 ]
    return inp_2

def get_model_path(model_id):
    with open(ICONFG_FILE) as f:
        confs = json.load(f)
        model_root = confs['models_root']
        models = confs['models']
        path = [ model["path"] for model in models if model["id"] == model_id]
        final_path =  os.path.join(model_root, path[0])
        return final_path

def replace_num_target_prefix(i_,num_map):
    num_tp = re.findall(patterns['p3']['regex'],i_['target_prefix'])
    try:
        for i in num_tp:
            replacement_tag =  [pair['tag'] for pair in num_map if str(pair['no.'])== i]
            if len(replacement_tag) > 0:
                replacement_tag = replacement_tag[0]
                i_['target_prefix'] = i_['target_prefix'].replace(i,replacement_tag)
        logger.info("tp after replacing numbers with tag: {}".format(i_['target_prefix']))
        return i_['target_prefix']
    except Exception as e:
        logger.error("Error in interavtive translation-replace_num_target_prefix:{}".format(e))
        return i_['target_prefix']
    