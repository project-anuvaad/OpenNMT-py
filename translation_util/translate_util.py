import json
import anuvada
import tools.sp_enc_dec as sp
import ancillary_functions_anuvaad.ancillary_functions as ancillary_functions
import ancillary_functions_anuvaad.sc_preface_handler as sc_preface_handler
import ancillary_functions_anuvaad.handle_date_url as date_url_util
from config.config import statusCode, benchmark_types, language_supported, file_location
from onmt.utils.logging import init_logger, logger
import os
from mongo_model import db, Benchmarks
import datetime
from onmt.translate import ServerModelError
import sys
from config import sentencepiece_model_loc as sp_model

def encode_translate_decode(i,translation_server,sp_encoder,sp_decoder):
    try:
        i['src'] = str(sp.encode_line(sp_encoder,i['src']))
        print("encoding:",i['src'])
        input_sw = i['src']
        translation, scores, n_best, times = translation_server.run([i])
        print("output from model:",translation[0])
        output_sw = translation[0]
        translation = sp.decode_line(sp_decoder,translation[0])
        print("decoded",translation)
        return translation,scores,input_sw,output_sw
    except Exception as e:
        # out['status'] = statusCode["SYSTEM_ERR"]
        print("error in encode_translate_decode function: {}".format(e))
        logger.info("Unexpected error: %s"% sys.exc_info()[0])


def from_en(inputs, translation_server):
        inputs = inputs
        out = {}
        tgt = list()
        pred_score = list()
        sentence_id = list()
        node_id = list()
        input_subwords = list()
        output_subwords = list()
        s_id = [0000]
        n_id = [0000]  
        try:
            for i in inputs:
                if all(v in i for v in ['s_id','n_id']):
                    s_id = [i['s_id']]
                    n_id = [i['n_id']]

                if  any(v not in i for v in ['src','id']):
                    out['status'] = statusCode["ID_OR_SRC_MISSING"]
                    return out

                i['src'] = i['src'].strip()
                if ancillary_functions.special_case_fits(i['src']):
                    logger.info("sentence fits in special case, returning accordingly and not going to model")
                    translation = ancillary_functions.handle_special_cases(i['src'],i['id'])
                    scores = [1]      
                else:
                    logger.info("translating using NMT-model:{}".format(i['id']))
                    # prefix,suffix, i['src'] = ancillary_functions.separate_alphanumeric_and_symbol(i['src'])
                    # print("prefix :{},suffix :{},i[src] :{}".format(prefix,suffix,i['src']))
                    if i['id'] == 1:
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_220519"],sp_model.english_hindi["HIN_220519"])
                        print("output from function:",translation)
                        translation = anuvada.indic_detokenizer(translation)
                    elif i['id'] == 8:
                        numbers = sc_preface_handler.get_numbers(i['src'])
                        i['src'] = sc_preface_handler.replace_numbers_with_hash(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["SC_PREFACE_ENG"],sp_model.english_hindi["SC_PREFACE_HIN"])
                        print("output from function:",translation)
                        translation = sc_preface_handler.replace_hash_with_original_number(translation,numbers)  
                    elif i['id'] == 7:  
                        "english-tamil"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_tamil["ENG_230919"],sp_model.english_tamil["TAM_230919"])
                        logger.info("decoded tamil: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 10:  
                        "english-gujrati"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_gujrati["ENG_100919"],sp_model.english_gujrati["GUJ_100919"])
                        logger.info("decoded gujrati: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 11:  
                        "english-bengali"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_bengali["ENG_120919"],sp_model.english_bengali["BENG_120919"])
                        logger.info("decoded bengali: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array) 
                    elif i['id'] == 12:  
                        "english-marathi"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_marathi["ENG_140919"],sp_model.english_marathi["MARATHI_140919"])
                        logger.info("decoded marathi: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] in [13,14]:  
                        "170919 eng-hin"
                        i['src'] = i['src'].lower()
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_170919"],sp_model.english_hindi["HIN_170919"])
                        logger.info("decoded hindi: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)                   

                    elif i['id'] == 15:  
                        "english-kannada"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_kannada["ENG_200919"],sp_model.english_kannada["KANNADA_200919"])
                        logger.info("decoded kannada: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 16:  
                        "english-telgu"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_telgu["ENG_200919"],sp_model.english_telgu["TELGU_200919"])
                        logger.info("decoded telgu: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 17:  
                        "english-malayalam"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_malayalam["ENG_200919"],sp_model.english_malayalam["MALAYALAM_200919"])
                        logger.info("decoded malayalam: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    elif i['id'] == 18:  
                        "english-punjabi"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src']) 
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_punjabi["ENG_200919"],sp_model.english_punjabi["PUNJABI_200919"])
                        logger.info("decoded punjabi: {}".format(translation))
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                    else:
                        logger.info("unsupported model id: {} for given english translation".format(i['id']))
                        logger.error("unsupported model id: {} for given english translation".format(i['id']))
                        translation = ""
                        input_sw = ""
                        output_sw = ""
                        scores = [0]
                        
                    
                    # translation = (prefix+" "+translation+" "+suffix).strip()
                print("here3")
                translation = ancillary_functions.replace_hindi_numbers(translation)
                tgt.append(translation)
                pred_score.append(scores[0])
                sentence_id.append(s_id[0])
                node_id.append(n_id[0])
                input_subwords.append(input_sw)
                output_subwords.append(output_sw)

            out['status'] = statusCode["SUCCESS"]
            out['response_body'] = [{"tgt": tgt[i],
                                     "s_id": sentence_id[i],"input_subwords": input_subwords[i], 
                                     "output_subwords":output_subwords[i],"n_id":node_id[i],"pred_score": pred_score[i]}
                    for i in range(len(tgt))]
        except ServerModelError as e:
            out['status'] = statusCode["SEVER_MODEL_ERR"]
            out['status']['errObj'] = str(e)
        except Exception as e:
            out['status'] = statusCode["SYSTEM_ERR"]
            print("error: {}".format(e))
            logger.info("Unexpected error: %s"% sys.exc_info()[0])    

        return out


def from_hindi(inputs, translation_server):
    inputs = inputs
    out = {}
    tgt = list()
    pred_score = list()
    sentence_id = list()
    node_id = list()
    input_subwords = list()
    output_subwords = list()
    s_id = [0000]
    n_id = [0000]

    try:
        for i in inputs:
            if 's_id'in i:
                s_id = [i['s_id']]
            else:
                s_id = [0000]   
            if  any(v not in i for v in ['src','id']):
                out['status'] = statusCode["ID_OR_SRC_MISSING"]
                return (out) 
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
                    return (out)      
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

    return (out)