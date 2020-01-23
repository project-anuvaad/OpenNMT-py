import sys
import os
from onmt.utils.logging import init_logger
import tools.indic_tokenize as hin_tokenizer
import tools.sp_enc_dec as sp
import datetime
import corpus.master_corpus_location as mcl
import corpus.scripts as corpus_scripts
import utils.training_utils.onmt_utils as onmt_utils
import utils.training_utils.pairwise_processing as pairwise_processing

date_now = datetime.datetime.now().strftime('%Y-%m-%d')
INTERMEDIATE_DATA_LOCATION = 'intermediate_data/'
TRAIN_DEV_TEST_DATA_LOCATION = 'data/'
NMT_MODEL_DIR = 'model/'
SENTENCEPIECE_MODEL_DIR = 'model/sentencepiece_models/'
TRAIN_LOG_FILE = 'available_models/anuvaad_training_log_file.txt'

logger = init_logger(TRAIN_LOG_FILE)


def english_hindi_experiments(model_type,eng_file,hindi_file,epoch,experiment_key):
    try:
        experiment_key = experiment_key or "default"
        logger.info("request for {} training on {}-{} and epoch:{} for exp:{}".format(model_type,eng_file,hindi_file,epoch,experiment_key))
        preprocessed_data = corpus_scripts.english_hindi_experiments(eng_file,hindi_file)
        preprocessed_data['experiment_key'] = experiment_key

        f_out = pairwise_processing.english_and_hindi(preprocessed_data)

        if model_type == "english-hindi":
            training_para = {'train_src':f_out['english_encoded_file'],'train_tgt':f_out['hindi_encoded_file'], \
                            'valid_src':f_out['english_dev_encoded_file'],"valid_tgt":f_out['hindi_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}
        elif model_type == "hindi-english":
            training_para = {'train_src':f_out['hindi_encoded_file'],'train_tgt':f_out['english_encoded_file'], \
                            'valid_src':f_out['hindi_dev_encoded_file'],"valid_tgt":f_out['english_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}                                      
        
        onmt_utils.onmt_train(training_para)
    except Exception as e:
        logger.error("error in english_hindi_experiments anuvaad script: {}".format(e))

def incremental_training(src_sp_model,tgt_sp_model,train_from_model):
    try:
        "in progress"
        print("h")
    except Exception as e:
        print(e)
        logger.info("error in english_hindi anuvaad script: {}".format(e))


def english_tamil(model_type,eng_file,tamil_file,epoch,experiment_key):
    try:
        experiment_key = experiment_key or "default"
        logger.info("request for {} training on {}-{} and epoch:{} for exp:{}".format(model_type,eng_file,tamil_file,epoch,experiment_key))
        preprocessed_data = corpus_scripts.english_tamil(eng_file,tamil_file)
        preprocessed_data['experiment_key'] = experiment_key

        f_out = pairwise_processing.english_and_tamil(preprocessed_data)

        if model_type == "english-tamil":
            training_para = {'train_src':f_out['english_encoded_file'],'train_tgt':f_out['tamil_encoded_file'], \
                            'valid_src':f_out['english_dev_encoded_file'],"valid_tgt":f_out['tamil_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}
        elif model_type == "tamil-english":
            training_para = {'train_src':f_out['tamil_encoded_file'],'train_tgt':f_out['english_encoded_file'], \
                            'valid_src':f_out['tamil_dev_encoded_file'],"valid_tgt":f_out['english_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}                                      
        
        onmt_utils.onmt_train(training_para)
    except Exception as e:
        logger.error("error in english_tamil_experiments anuvaad script: {}".format(e))
   

def english_gujarati(model_type,eng_file,gujarati_file,epoch,experiment_key):
    try:
        experiment_key = experiment_key or "default"
        logger.info("request for {} training on {}-{} and epoch:{} for exp:{}".format(model_type,eng_file,gujarati_file,epoch,experiment_key))
        preprocessed_data = corpus_scripts.english_gujarati(eng_file,gujarati_file)
        preprocessed_data['experiment_key'] = experiment_key

        f_out = pairwise_processing.english_and_gujarati(preprocessed_data)

        if model_type == "english-gujarati":
            training_para = {'train_src':f_out['english_encoded_file'],'train_tgt':f_out['gujarati_encoded_file'], \
                            'valid_src':f_out['english_dev_encoded_file'],"valid_tgt":f_out['gujarati_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}
        elif model_type == "gujarati-english":
            training_para = {'train_src':f_out['gujarati_encoded_file'],'train_tgt':f_out['english_encoded_file'], \
                            'valid_src':f_out['gujarati_dev_encoded_file'],"valid_tgt":f_out['english_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}                                      
        
        logger.info("starting onmt_train for :{}".format(model_type))
        onmt_utils.onmt_train(training_para)
    except Exception as e:
        logger.error("error in english_gujarati anuvaad script: {}".format(e))
    

def english_bengali(model_type,eng_file,bengali_file,epoch,experiment_key):
    try:
        experiment_key = experiment_key or "default"
        logger.info("request for {} training on {}-{} and epoch:{} for exp:{}".format(model_type,eng_file,bengali_file,epoch,experiment_key))
        preprocessed_data = corpus_scripts.english_bengali(eng_file,bengali_file)
        preprocessed_data['experiment_key'] = experiment_key

        f_out = pairwise_processing.english_and_bengali(preprocessed_data)

        if model_type == "english-bengali":
            training_para = {'train_src':f_out['english_encoded_file'],'train_tgt':f_out['bengali_encoded_file'], \
                            'valid_src':f_out['english_dev_encoded_file'],"valid_tgt":f_out['bengali_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}
        elif model_type == "bengali-english":
            training_para = {'train_src':f_out['bengali_encoded_file'],'train_tgt':f_out['english_encoded_file'], \
                            'valid_src':f_out['bengali_dev_encoded_file'],"valid_tgt":f_out['english_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}                                      
        
        logger.info("starting onmt_train for :{}".format(model_type))
        onmt_utils.onmt_train(training_para)
    except Exception as e:
        logger.error("error in english_bengali anuvaad script: {}".format(e)) 
   

def english_marathi(model_type,eng_file,marathi_file,epoch,experiment_key):
    try:
        experiment_key = experiment_key or "default"
        logger.info("request for {} training on {}-{} and epoch:{} for exp:{}".format(model_type,eng_file,marathi_file,epoch,experiment_key))
        preprocessed_data = corpus_scripts.english_marathi(eng_file,marathi_file)
        preprocessed_data['experiment_key'] = experiment_key

        f_out = pairwise_processing.english_and_marathi(preprocessed_data)

        if model_type == "english-marathi":
            training_para = {'train_src':f_out['english_encoded_file'],'train_tgt':f_out['marathi_encoded_file'], \
                            'valid_src':f_out['english_dev_encoded_file'],"valid_tgt":f_out['marathi_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}
        elif model_type == "marathi-english":
            training_para = {'train_src':f_out['marathi_encoded_file'],'train_tgt':f_out['english_encoded_file'], \
                            'valid_src':f_out['marathi_dev_encoded_file'],"valid_tgt":f_out['english_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}                                      
        
        logger.info("starting onmt_train for :{}".format(model_type))
        onmt_utils.onmt_train(training_para)
    except Exception as e:
        logger.error("error in english_marathi anuvaad script: {}".format(e)) 
   

def english_kannada(model_type,eng_file,kannada_file,epoch,experiment_key): 
    try:
        experiment_key = experiment_key or "default"
        logger.info("request for {} training on {}-{} and epoch:{} for exp:{}".format(model_type,eng_file,kannada_file,epoch,experiment_key))
        preprocessed_data = corpus_scripts.english_kannada(eng_file,kannada_file)
        preprocessed_data['experiment_key'] = experiment_key

        f_out = pairwise_processing.english_and_kannada(preprocessed_data)

        if model_type == "english-kannada":
            training_para = {'train_src':f_out['english_encoded_file'],'train_tgt':f_out['kannada_encoded_file'], \
                            'valid_src':f_out['english_dev_encoded_file'],"valid_tgt":f_out['kannada_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}
        elif model_type == "kannada-english":
            training_para = {'train_src':f_out['kannada_encoded_file'],'train_tgt':f_out['english_encoded_file'], \
                            'valid_src':f_out['kannada_dev_encoded_file'],"valid_tgt":f_out['english_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}                                      
        
        logger.info("starting onmt_train for :{}".format(model_type))
        onmt_utils.onmt_train(training_para)
    except Exception as e:
        logger.error("error in english_kannada anuvaad script: {}".format(e))
    

def english_telugu(model_type,eng_file,telugu_file,epoch,experiment_key):
    try:
        experiment_key = experiment_key or "default"
        logger.info("request for {} training on {}-{} and epoch:{} for exp:{}".format(model_type,eng_file,telugu_file,epoch,experiment_key))
        preprocessed_data = corpus_scripts.english_telugu(eng_file,telugu_file)
        preprocessed_data['experiment_key'] = experiment_key

        f_out = pairwise_processing.english_and_telugu(preprocessed_data)

        if model_type == "english-telugu":
            training_para = {'train_src':f_out['english_encoded_file'],'train_tgt':f_out['telugu_encoded_file'], \
                            'valid_src':f_out['english_dev_encoded_file'],"valid_tgt":f_out['telugu_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}
        elif model_type == "telugu-english":
            training_para = {'train_src':f_out['telugu_encoded_file'],'train_tgt':f_out['english_encoded_file'], \
                            'valid_src':f_out['telugu_dev_encoded_file'],"valid_tgt":f_out['english_dev_encoded_file'], \
                            'nmt_processed_data':f_out['nmt_processed_data'],'nmt_model_path':f_out['nmt_model_path'],'epoch':epoch}                                      
        
        logger.info("starting onmt_train for :{}".format(model_type))
        onmt_utils.onmt_train(training_para)
    except Exception as e:
        logger.error("error in english_telugu anuvaad script: {}".format(e)) 
   

def english_malayalam(): 
    try:
       print("hi") 
    except Exception as e:
        print(e)
        logger.info("error in english_malayalam anuvaad script: {}".format(e))

def english_punjabi(): 
    try:
        print("hi")
    except Exception as e:
        print(e)
        logger.info("error in english_punjabi anuvaad script: {}".format(e))


if __name__ == '__main__':
    try:
        if sys.argv[1] in ["english-tamil","tamil-english"]:
            english_tamil(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
        elif sys.argv[1] in ["english-gujarati","gujarati-english"]:
            english_gujarati(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
        elif sys.argv[1] in ["english-bengali","bengali-english"]:
            english_bengali(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
        elif sys.argv[1] in ["english-marathi","marathi-english"]:
            english_marathi(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
        elif sys.argv[1] in ["english-kannada","kannada-english"]:
            english_kannada(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5]) 
        elif sys.argv[1] in ["english-telugu","telugu-english"]:
            english_telugu(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])
        elif sys.argv[1] == "english-malayalam":
            english_malayalam() 
        elif sys.argv[1] == "english-punjabi":
            english_punjabi() 
        elif sys.argv[1] in ["english-hindi","hindi-english"]:
            english_hindi_experiments(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5])  
        elif sys.argv[1] == "incremental_training":
            incremental_training(sys.argv[2],sys.argv[3],sys.argv[4])                                                  
        else:
            logger.error("Invalid request type-{}".format(sys.argv[1]))
    except Exception as e:
        logger.error("Error while calling Anuvaad Script-{}, Error:{}".format(sys.argv[1],e))
    
