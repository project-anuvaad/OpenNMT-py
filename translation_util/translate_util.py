import json
import tools.sp_enc_dec as sp
import ancillary_functions_anuvaad.sentence_processor as sentence_processor
import ancillary_functions_anuvaad.ancillary_functions as ancillary_functions
import ancillary_functions_anuvaad.sc_preface_handler as sc_preface_handler
import ancillary_functions_anuvaad.handle_date_url as date_url_util
import ancillary_functions_anuvaad.output_cleaner as oc
from config.config import statusCode, benchmark_types, language_supported, file_location
from onmt.utils.logging import init_logger, logger,log_with_request_info,LOG_TAGS
import os
import datetime
from onmt.translate import ServerModelError
import sys
from config import sentencepiece_model_loc as sp_model
from config.regex_patterns import patterns

def encode_translate_decode(i,translation_server,sp_encoder,sp_decoder):
    try:
        logger.info("Inside encode_translate_decode function")
        i['src'] = str(sp.encode_line(sp_encoder,i['src']))
        logger.info("SP encoded sent: %s"%i['src'])
        input_sw = i['src']
        translation, scores, n_best, times = translation_server.run([i])
        logger.info("output from model: %s"%translation[0])
        output_sw = translation[0]
        translation = sp.decode_line(sp_decoder,translation[0])
        logger.info("SP decoded sent: %s"%translation)
        return translation,scores,input_sw,output_sw
    except ServerModelError as e:
        logger.error("ServerModelError error in encode_translate_decode: {} and {}".format(e,sys.exc_info()[0]))
        raise
        
    except Exception as e:
        logger.error("Unexpexcted error in encode_translate_decode: {} and {}".format(e,sys.exc_info()[0]))
        raise


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
                    logger.info("either id or src missing in some input")
                    return out

                logger.info("input sentences:{}".format(i['src']))
                i['src'] = i['src'].strip()
                if ancillary_functions.special_case_fits(i['src']):
                    logger.info("sentence fits in special case, returning accordingly and not going to model")
                    translation = ancillary_functions.handle_special_cases(i['src'],i['id'])
                    scores = [1] 
                    input_sw,output_sw = "",""     
                else:
                    logger.info("translating using NMT-model:{}".format(i['id']))
                    logger.info("translating this sentences:{}".format(i['src']))
                    # prefix,suffix, i['src'] = ancillary_functions.separate_alphanumeric_and_symbol(i['src'])
                    if i['id'] == 1:
                        i['src'] = sentence_processor.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_220519"],sp_model.english_hindi["HIN_220519"])
                        translation = sentence_processor.indic_detokenizer(translation)
                        logger.info("final output from model-1: {}".format(translation))  
                    else:
                        logger.info("unsupported model id: {} for given english translation".format(i['id']))
                        logger.error("unsupported model id: {} for given english translation".format(i['id']))
                        translation,input_sw,output_sw,scores = "","","",[0]
                        
                    
                    # translation = (prefix+" "+translation+" "+suffix).strip()
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
            out['status']['why'] = str(e)
            logger.error("ServerModelError error in TRANSLATE_UTIL-FROM_ENGLISH: {} and {}".format(e,sys.exc_info()[0]))
        except Exception as e:
            out['status'] = statusCode["SYSTEM_ERR"]
            logger.error("Unexpected error in translate_util from_eng function: %s and %s"% (e,sys.exc_info()[0]))    

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
            if all(v in i for v in ['s_id','n_id']):
                s_id = [i['s_id']]
                n_id = [i['n_id']]  
                
            if  any(v not in i for v in ['src','id']):
                out['status'] = statusCode["ID_OR_SRC_MISSING"]
                logger.info("either id or src missing in some input")
                return (out) 

            logger.info("input sentences:{}".format(i['src']))    
            i['src'] = i['src'].strip()    
            if i['id'] == 3:
                logger.info("translating using the first model")
                translation, scores, n_best, times = translation_server.run([i])
                translation = translation[0]
                input_sw,output_sw = "",""   

            else:
                if i['id'] == 2:
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["HIN_220519"],sp_model.english_hindi["ENG_220519"])
                    translation = sentence_processor.moses_detokenizer(translation)
                    translation = sentence_processor.detruecaser(translation)
                 
                else:
                    logger.info("unsupported model id: {} for given hindi input for translation".format(i['id']))
                    translation = ""
                    input_sw,output_sw = "",""
                    scores = [0]         
                
            tgt.append(translation)
            pred_score.append(scores[0])
            sentence_id.append(s_id[0])
            node_id.append(n_id[0])
            input_subwords.append(input_sw)
            output_subwords.append(output_sw)

        out['status'] = statusCode["SUCCESS"]
        out['response_body'] = [{"tgt": tgt[i],
                "pred_score": pred_score[i], "s_id": sentence_id[i],"input_subwords": input_subwords[i],
                "output_subwords":output_subwords[i],"n_id":node_id[i]}
                for i in range(len(tgt))]
    except ServerModelError as e:
        out['status'] = statusCode["SEVER_MODEL_ERR"]
        out['status']['why'] = str(e)
        logger.error("ServerModelError error in TRANSLATE_UTIL-FROM_HINDI: {} and {}".format(e,sys.exc_info()[0]))
    except Exception as e:
        out['status'] = statusCode["SYSTEM_ERR"]
        out['status']['why'] = str(e)
        logger.error("Unexpected error:%s and %s"% (e,sys.exc_info()[0]))   

    return (out)


def translate_func(inputs, translation_server):

    inputs = inputs
    out = {}
    pred_score = list()
    sentence_id,node_id = list(),list()
    input_subwords,output_subwords = list(),list()
    i_src,tgt = list(),list()
    tagged_tgt,tagged_src = list(),list()
    s_id,n_id = [0000],[0000]

    try:
        for i in inputs:
            logger.info(log_with_request_info(i.get("s_id"),LOG_TAGS["input"],i))
            if all(v in i for v in ['s_id','n_id']):
                s_id = [i['s_id']]
                n_id = [i['n_id']]  
                
            if  any(v not in i for v in ['src','id']):
                out['status'] = statusCode["ID_OR_SRC_MISSING"]
                out['response_body'] = []
                logger.info("either id or src missing in some input")
                return (out) 

            logger.info("input sentences:{}".format(i['src'])) 
            i_src.append(i['src'])   
            i['src'] = i['src'].strip()
            if ancillary_functions.special_case_fits(i['src']):
                logger.info("sentence fits in special case, returning accordingly and not going to model")
                translation = ancillary_functions.handle_special_cases(i['src'],i['id'])
                scores = [1] 
                input_sw,output_sw,tag_tgt,tag_src = "","",translation,i['src']

            else:
                logger.info("translating using NMT-model:{}".format(i['id']))
                # prefix,suffix, i['src'] = ancillary_functions.separate_alphanumeric_and_symbol(i['src'])
                prefix, i['src'] = ancillary_functions.prefix_handler(i['src'])
                i['src'],date_original,url_original,num_array,num_map = date_url_util.tag_number_date_url_1(i['src'])
                tag_src = (prefix +" "+ i['src']).lstrip() 
                if i['id'] == 5:
                    "hi-en exp-1"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.hindi_english["HIN_EXP_1_291019"],sp_model.hindi_english["ENG_EXP_1_291019"])
                    translation = sentence_processor.moses_detokenizer(translation)
                elif i['id'] == 6:
                    "hi-en_exp-2 05-05-20"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.hindi_english["HIN_EXP_2_050520"],sp_model.hindi_english["ENG_EXP_2_050520"])
                    translation = sentence_processor.moses_detokenizer(translation)

                elif i['id'] == 7:  
                    "english-tamil"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_tamil["ENG_230919"],sp_model.english_tamil["TAM_230919"])
                elif i['id'] == 10:  
                    "english-gujrati"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_gujarati["ENG_100919"],sp_model.english_gujarati["GUJ_100919"])
                    translation = translation.replace("ન્યાય માટે Accessક્સેસને","ન્યાયની પહોંચને")
                elif i['id'] == 11:  
                    "english-bengali"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_bengali["ENG_120919"],sp_model.english_bengali["BENG_120919"])
                elif i['id'] == 12:  
                    "english-marathi"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_marathi["ENG_140919"],sp_model.english_marathi["MARATHI_140919"])               

                elif i['id'] == 15:  
                    "english-kannada"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_kannada["ENG_200919"],sp_model.english_kannada["KANNADA_200919"])
                    translation = translation.replace("uc","")
                elif i['id'] == 16:  
                    "english-telgu"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_telugu["ENG_200919"],sp_model.english_telugu["TELGU_200919"])
                elif i['id'] == 17:  
                    "english-malayalam"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_malayalam["ENG_200919"],sp_model.english_malayalam["MALAYALAM_200919"])
                elif i['id'] == 18:  
                    "english-punjabi"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_punjabi["ENG_200919"],sp_model.english_punjabi["PUNJABI_200919"])
                elif i['id'] == 21:  
                    "exp-1 BPE model with varying vocab size 15k for both hindi and english +tokenization"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_1"],sp_model.english_hindi["HIN_EXP_1"])                      
                    translation = sentence_processor.indic_detokenizer(translation)  
                elif i['id'] == 30:
                    "25/10/2019 experiment 10, Old data + dictionary,BPE-24k, nolowercasing,pretok,shuffling,50k nmt"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_10"],sp_model.english_hindi["HIN_EXP_10"])                      
                    translation = sentence_processor.indic_detokenizer(translation)   
                elif i['id'] == 32:
                    "29/10/2019 Exp-12: old_data_original+lc_cleaned+ ik names translated from google(100k)+shabdkosh(appended 29k new),BPE-24K,50knmt,shuff,pretok"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_12"],sp_model.english_hindi["HIN_EXP_12"])                      
                    translation = sentence_processor.indic_detokenizer(translation)
                elif i['id'] == 54:
                    "29-30/10/19Exp-5.4: -data same as 5.1 exp...old data+ india kanoon 830k(including 1.5 lakhs names n no learned counsel)+72192k shabkosh, BPE 24k, nolowercasing,pretok,shuffling"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_5.4"],sp_model.english_hindi["HIN_EXP_5.4"])                      
                    translation = sentence_processor.indic_detokenizer(translation)
                elif i['id'] == 42:  
                    "english-marathi exp-2"
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_marathi["ENG_071119"],sp_model.english_marathi["MARATHI_071119"])    
                elif i['id'] == 56:
                    "09/12/19-Exp-5.6:" 
                    if i['src'].isupper():
                        i['src'] = i['src'].title()
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_5.6"],sp_model.english_hindi["HIN_EXP_5.6"])                      
                    translation = sentence_processor.indic_detokenizer(translation)
                elif i['id'] == 8:
                    "ta-en 1st"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_tamil["TAM_090120"],sp_model.english_tamil["ENG_090120"])
                    translation = sentence_processor.moses_detokenizer(translation)  
                elif i['id'] == 43:
                    "mr-en 1st"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_marathi["MARATHI_270120"],sp_model.english_marathi["ENG_270120"])
                    translation = sentence_processor.moses_detokenizer(translation)  
                elif i['id'] == 44:
                    "eng-mr-3rd"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_marathi["ENG_060220"],sp_model.english_marathi["MARATHI_060220"])
                    translation = sentence_processor.indic_detokenizer(translation)         
                elif i['id'] == 45:
                    "en-ta 4th"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_tamil["ENG_080220"],sp_model.english_tamil["TAM_080220"])
                    translation = sentence_processor.indic_detokenizer(translation)  
                elif i['id'] == 46:
                    "ta-en 2nd"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_tamil["TAM_100220"],sp_model.english_tamil["ENG_100220"])
                    translation = sentence_processor.moses_detokenizer(translation)  
                elif i['id'] == 47:
                    "en-kn 2nd"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_kannada["ENG_100220"],sp_model.english_kannada["KANNADA_100220"])
                    translation = sentence_processor.indic_detokenizer(translation) 
                elif i['id'] == 48:
                    "kn-en 1st"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_kannada["KANNADA_100220"],sp_model.english_kannada["ENG_100220"])
                    translation = sentence_processor.moses_detokenizer(translation)
                elif i['id'] == 49:
                    "en-tel 2nd"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_telugu["ENG_120220"],sp_model.english_telugu["TELUGU_120220"])
                    translation = sentence_processor.indic_detokenizer(translation) 
                elif i['id'] == 50:
                    "tel-en 1st"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_telugu["TELUGU_120220"],sp_model.english_telugu["ENG_120220"])
                    translation = sentence_processor.moses_detokenizer(translation)
                elif i['id'] == 51:
                    "en-guj 2nd"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_gujarati["ENG_140220"],sp_model.english_gujarati["GUJ_140220"])
                    translation = sentence_processor.indic_detokenizer(translation) 
                elif i['id'] == 52:
                    "guj-en 1st"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_gujarati["GUJ_140220"],sp_model.english_gujarati["ENG_140220"])
                    translation = sentence_processor.moses_detokenizer(translation)
                elif i['id'] == 53:
                    "en-punjabi 2nd"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_punjabi["ENG_160220"],sp_model.english_punjabi["PUNJABI_160220"])
                    translation = sentence_processor.indic_detokenizer(translation) 
                elif i['id'] == 55:
                    "punjabi-en 1st"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_punjabi["PUNJABI_160220"],sp_model.english_punjabi["ENG_160220"])
                    translation = sentence_processor.moses_detokenizer(translation)
                elif i['id'] == 57:
                    "en-bengali 2nd"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_bengali["ENG_180220"],sp_model.english_bengali["BENG_180220"])
                    translation = sentence_processor.indic_detokenizer(translation) 
                elif i['id'] == 58:
                    "bengali-en 1st"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_bengali["BENG_180220"],sp_model.english_bengali["ENG_180220"])
                    translation = sentence_processor.moses_detokenizer(translation)
                elif i['id'] == 59:
                    "en-malay 2nd"
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_malayalam["ENG_210220"],sp_model.english_malayalam["MALAYALAM_210220"])
                    translation = sentence_processor.indic_detokenizer(translation) 
                elif i['id'] == 60:
                    "malay-en 1st"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_malayalam["MALAYALAM_210220"],sp_model.english_malayalam["ENG_210220"])
                    translation = sentence_processor.moses_detokenizer(translation)
                elif i['id'] == 61:
                    "ta-to-en 3rd"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_tamil["TAM_280220"],sp_model.english_tamil["ENG_280220"])
                    translation = sentence_processor.moses_detokenizer(translation) 
                elif i['id'] == 62:
                    "mr-to-en 2nd"
                    i['src'] = sentence_processor.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_marathi["MARATHI_280220"],sp_model.english_marathi["ENG_280220"])
                    translation = sentence_processor.moses_detokenizer(translation)
                elif i['id'] == 63:
                    "en-hi exp-13 09-03-20"  
                    i['src'] = sentence_processor.moses_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_13"],sp_model.english_hindi["HIN_EXP_13"])                      
                    translation = sentence_processor.indic_detokenizer(translation)                                                     
                else:
                    logger.info("Unsupported model id: {} for given input".format(i['id']))
                    raise Exception("Unsupported Model ID - id: {} for given input".format(i['id']))      

                # translation = (prefix+" "+translation+" "+suffix).strip()
                translation = (prefix+" "+translation).lstrip()
                translation = translation.replace("▁"," ")
                translation = date_url_util.regex_pass(translation,[patterns['p8'],patterns['p9'],patterns['p4'],patterns['p5'],
                                            patterns['p6'],patterns['p7']])
                tag_tgt = translation                            
                translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                translation = oc.cleaner(tag_src,translation,i['id'])
            logger.info("trans_function-experiment-{} output: {}".format(i['id'],translation))   
            logger.info(log_with_request_info(i.get("s_id"),LOG_TAGS["output"],translation)) 
            tgt.append(translation)
            pred_score.append(scores[0])
            sentence_id.append(s_id[0]), node_id.append(n_id[0])
            input_subwords.append(input_sw), output_subwords.append(output_sw)
            tagged_tgt.append(tag_tgt), tagged_src.append(tag_src)

        out['status'] = statusCode["SUCCESS"]
        out['response_body'] = [{"tgt": tgt[i],
                "pred_score": pred_score[i], "s_id": sentence_id[i],"input_subwords": input_subwords[i],
                "output_subwords":output_subwords[i],"n_id":node_id[i],"src":i_src[i],
                "tagged_tgt":tagged_tgt[i],"tagged_src":tagged_src[i]}
                for i in range(len(tgt))]
    except ServerModelError as e:
        out['status'] = statusCode["SEVER_MODEL_ERR"]
        out['status']['why'] = str(e)
        out['response_body'] = []
        logger.error("ServerModelError error in TRANSLATE_UTIL-translate_func: {} and {}".format(e,sys.exc_info()[0]))
    except Exception as e:
        out['status'] = statusCode["SYSTEM_ERR"]
        out['status']['why'] = str(e)
        out['response_body'] = []
        logger.error("Unexpected error:%s and %s"% (e,sys.exc_info()[0]))   

    return (out)

