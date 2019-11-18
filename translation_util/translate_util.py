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
        return "",[0],"",""
        
    except Exception as e:
        logger.error("Unexpexcted error in encode_translate_decode: {} and {}".format(e,sys.exc_info()[0]))
        return "",[0],"",""


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
                    # print("prefix :{},suffix :{},i[src] :{}".format(prefix,suffix,i['src']))
                    if i['id'] == 1:
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_220519"],sp_model.english_hindi["HIN_220519"])
                        translation = anuvada.indic_detokenizer(translation)
                        logger.info("final output from model-1: {}".format(translation))
                    elif i['id'] == 8:
                        numbers = sc_preface_handler.get_numbers(i['src'])
                        i['src'] = sc_preface_handler.replace_numbers_with_hash(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["SC_PREFACE_ENG"],sp_model.english_hindi["SC_PREFACE_HIN"])
                        translation = sc_preface_handler.replace_hash_with_original_number(translation,numbers) 
                        logger.info("final output from chota model-8: {}".format(translation)) 
                    elif i['id'] == 7:  
                        "english-tamil"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_tamil["ENG_230919"],sp_model.english_tamil["TAM_230919"])                      
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                        logger.info("final output tamil: {}".format(translation))
                    elif i['id'] == 10:  
                        "english-gujrati"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_gujrati["ENG_100919"],sp_model.english_gujrati["GUJ_100919"])                       
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                        translation = translation.replace("ન્યાય માટે Accessક્સેસને","ન્યાયની પહોંચને")
                        logger.info("final output gujrati: {}".format(translation))
                    elif i['id'] == 11:  
                        "english-bengali"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_bengali["ENG_120919"],sp_model.english_bengali["BENG_120919"])                      
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array) 
                        logger.info("final output bengali: {}".format(translation))
                    elif i['id'] == 12:  
                        "english-marathi"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_marathi["ENG_140919"],sp_model.english_marathi["MARATHI_140919"])                        
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                        logger.info("final output marathi: {}".format(translation))
                    elif i['id'] in [13,14]:  
                        "170919 eng-hin"
                        i['src'] = i['src'].lower()
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_170919"],sp_model.english_hindi["HIN_170919"])                      
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("final output-13 hindi: {}".format(translation))                 

                    elif i['id'] == 15:  
                        "english-kannada"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_kannada["ENG_200919"],sp_model.english_kannada["KANNADA_200919"])                        
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                        translation = translation.replace("uc","")
                        logger.info("final output kannada: {}".format(translation))
                    elif i['id'] == 16:  
                        "english-telgu"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_telgu["ENG_200919"],sp_model.english_telgu["TELGU_200919"])                       
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                        logger.info("final output telgu: {}".format(translation))
                    elif i['id'] == 17:  
                        "english-malayalam"
                        i['src'] = i['src'].replace("litigants struggle","litigants struggling").replace("don't know","do not know")
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_malayalam["ENG_200919"],sp_model.english_malayalam["MALAYALAM_200919"])                       
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                        logger.info("final output malayalam: {}".format(translation))
                    elif i['id'] == 18:  
                        "english-punjabi"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src']) 
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_punjabi["ENG_200919"],sp_model.english_punjabi["PUNJABI_200919"])                       
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                        translation = translation.replace("infrastructureਾਂਚਾ","ਢਾਂਚਾ")
                        logger.info("final output punjabi: {}".format(translation))
                    elif i['id'] == 20:  
                        "180919 model, no tokenization,handling num,date,url+augmented data"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_180919"],sp_model.english_hindi["HIN_180919"])                      
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("180919 model output: {}".format(translation))
                    elif i['id'] == 21:  
                        "exp-1 BPE model with varying vocab size 15k for both hindi and english +tokenization"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_1"],sp_model.english_hindi["HIN_EXP_1"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-1 output: {}".format(translation))
                    elif i['id'] == 22:  
                        "experiment-2-Unigram model on existing corpus, and same params 10k +tokenization,4096"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_2"],sp_model.english_hindi["HIN_EXP_2"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-2 output: {}".format(translation)) 
                    elif i['id'] == 23:  
                        "experiment-3-BPE model, lowercasing all input while training and also during translation +tokenization, 4096"
                        i['src'] = i['src'].lower()
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_3"],sp_model.english_hindi["HIN_EXP_3"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-3 output: {}".format(translation)) 
                    elif i['id'] == 24:  
                        "experiment-4-Word based model + tokenization +6000 on 1gpu +all lowercasing"
                        i['src'] = i['src'].lower()
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation, scores, n_best, times = translation_server.run([i])
                        translation = anuvada.indic_detokenizer(translation[0])
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-4 output: {}".format(translation)) 
                        input_sw,output_sw = "",""
                    elif i['id'] == 26:  
                        "experiment-6 Unigram-10k +tok+ all lowercaseing +2*4096"
                        i['src'] = i['src'].lower()
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_6"],sp_model.english_hindi["HIN_EXP_6"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-6 output: {}".format(translation)) 
                    elif i['id'] == 27:
                        "exp-7 bpe-24k iit b+lc+etc. +pre-tokenisation 1*6000 my 1 gpu machine  12/10/19"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_7"],sp_model.english_hindi["HIN_EXP_7"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-7 output: {}".format(translation))  
                    elif i['id'] in [28,19]:
                        "Exp-8: Unigram(15k)+pre-tokenization ,IITB original(b+lc+etc.) new 1 gpu machine 12/10/19"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_8"],sp_model.english_hindi["HIN_EXP_8"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-8 output: {}".format(translation))   
                    elif i['id'] in [25,52,53]:
                        "18/10/19 Exp -5.1 and 5.2 -old data+ india kanoon 854517k(including 1.5 lakhs names)+72192k shabkosh, BPE 24k, nolowercasing,pretok,shuffling"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_5"],sp_model.english_hindi["HIN_EXP_5"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-{} output: {}".format(i['id'],translation)) 
                    elif i['id'] == 29:
                        "18/10/19 Exp -9 old data + india kanoon 800k(including 1.5 lakhs names)+74k shabkosh, unigram 24k, nolowercasing,pretok,shuffling,50k"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_9"],sp_model.english_hindi["HIN_EXP_9"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-9 output: {}".format(translation)) 
                    elif i['id'] == 30:
                        "25/10/2019 experiment 10, Old data + dictionary,BPE-24k, nolowercasing,pretok,shuffling,50k nmt"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_10"],sp_model.english_hindi["HIN_EXP_10"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-{} output: {}".format(i['id'],translation))   
                    elif i['id'] == 31:
                        "25/10/2019 experiment 11 Old data + dictionary+india kanoon cleaned_curated+lc_cleaned+BPE-24k, nolowercasing,pretok,shuffling,50knmt"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_11"],sp_model.english_hindi["HIN_EXP_11"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-{} output: {}".format(i['id'],translation))  
                    elif i['id'] == 32:
                        "29/10/2019 Exp-12: old_data_original+lc_cleaned+ ik names translated from google(100k)+shabdkosh(appended 29k new),BPE-24K,50knmt,shuff,pretok"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_12"],sp_model.english_hindi["HIN_EXP_12"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment-{} output: {}".format(i['id'],translation)) 
                    elif i['id'] == 54:
                        "29-30/10/19Exp-5.4: -data same as 5.1 exp...old data+ india kanoon 830k(including 1.5 lakhs names n no learned counsel)+72192k shabkosh, BPE 24k, nolowercasing,pretok,shuffling"                        
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        i['src'] = anuvada.moses_tokenizer(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["ENG_EXP_5.4"],sp_model.english_hindi["HIN_EXP_5.4"])                      
                        translation = anuvada.indic_detokenizer(translation)
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                        logger.info("experiment5.4-{} output: {}".format(i['id'],translation)) 
                    elif i['id'] == 42:  
                        "english-marathi exp-2"
                        i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                        translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_marathi["ENG_071119"],sp_model.english_marathi["MARATHI_071119"])                        
                        translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)
                        logger.info("final output marathi exp-2: {}".format(translation))                                    
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
            out['status']['errObj'] = str(e)
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
                    i['src'] = anuvada.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.english_hindi["HIN_220519"],sp_model.english_hindi["ENG_220519"])
                    translation = anuvada.moses_detokenizer(translation)
                    translation = anuvada.detruecaser(translation)
                elif i['id'] == 5:
                    i['src'],date_original,url_original,num_array = date_url_util.tag_number_date_url_1(i['src'])
                    i['src'] = anuvada.indic_tokenizer(i['src'])
                    translation,scores,input_sw,output_sw = encode_translate_decode(i,translation_server,sp_model.hindi_english["HIN_EXP_1_291019"],sp_model.hindi_english["ENG_EXP_1_291019"])
                    translation = anuvada.moses_detokenizer(translation)
                    translation = date_url_util.replace_tags_with_original_1(translation,date_original,url_original,num_array)  
                    logger.info("experiment-{} output: {}".format(i['id'],translation)) 
                elif i['id'] == 4:
                    i['src'] = anuvada.indic_tokenizer(i['src'])
                    i['src'] = anuvada.apply_bpe('codesSrc1005.bpe',i['src'])
                    input_sw,output_sw = i['src'],""
                    translation, scores, n_best, times = translation_server.run([i])
                    translation = anuvada.decode_bpe(translation[0])
                    translation = anuvada.moses_detokenizer(translation)
                    translation = anuvada.detruecaser(translation)
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
        out['status']['errObj'] = str(e)
        logger.error("ServerModelError error in TRANSLATE_UTIL-FROM_HINDI: {} and {}".format(e,sys.exc_info()[0]))
    except Exception as e:
        out['status'] = statusCode["SYSTEM_ERR"]
        out['status']['errObj'] = str(e)
        logger.error("Unexpected error:%s and %s"% (e,sys.exc_info()[0]))   

    return (out)