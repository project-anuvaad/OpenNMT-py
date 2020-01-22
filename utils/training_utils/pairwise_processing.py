"Pairwise preprocessing aims to do task which are common across two language pair."
"That is it doesn't matter which way we are training, this preprocessing is one time and works for both"

"***************InProgress*************"

import sys
import os
from onmt.utils.logging import init_logger
import tools.indic_tokenize as hin_tokenizer
import tools.sp_enc_dec as sp
import datetime
import corpus.master_corpus_location as mcl

date_now = datetime.datetime.now().strftime('%Y-%m-%d')
INTERMEDIATE_DATA_LOCATION = 'intermediate_data/'
TRAIN_DEV_TEST_DATA_LOCATION = 'data/'
NMT_MODEL_DIR = 'model/'
SENTENCEPIECE_MODEL_DIR = 'model/sentencepiece_models/'
TRAIN_LOG_FILE = 'available_models/anuvaad_training_log_file.txt'

logger = init_logger(TRAIN_LOG_FILE)


def english_and_tamil(inputs):
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_tamil = 'tamil-{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'enTa-{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_tamil')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_tamil')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_tamil','model_en-ta-{}_{}-model'.format(experiment_key,date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_tamil'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_tamil'))
            logger.info("folder created at {}".format(model_intermediate_folder))
        
        tamil_tokenized_file = os.path.join(model_intermediate_folder, 'tamil_train_tok'+unique_id+'.txt')
        tamil_dev_tokenized_file = os.path.join(model_intermediate_folder, 'tamil_dev_tok'+unique_id+'.txt')
        tamil_test_tokenized_file = os.path.join(model_intermediate_folder, 'tamil_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        tamil_encoded_file = os.path.join(model_master_train_folder, 'tamil_train_final'+unique_id+'.txt')
        tamil_dev_encoded_file = os.path.join(model_master_train_folder, 'tamil_dev_final'+unique_id+'.txt')
        tamil_test_encoded_file = os.path.join(model_master_train_folder, 'tamil_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-tamil pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} ta'.format(inputs['TAMIL_TRAIN_FILE'], tamil_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} ta'.format(inputs['DEV_TAMIL'], tamil_dev_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} ta'.format(mcl.english_tamil['TEST_TAMIL'], tamil_test_tokenized_file))
        logger.info("Eng-tamil pairwise preprocessing, tamil train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_tamil['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-tamil pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(tamil_tokenized_file,sp_model_prefix_tamil, 24000, 'bpe')
        logger.info("sentencepiece model tamil trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),tamil_tokenized_file,tamil_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),tamil_dev_tokenized_file,tamil_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),tamil_test_tokenized_file,tamil_test_encoded_file)
        logger.info("tamil-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(tamil_tokenized_file,tamil_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['TAMIL_TRAIN_FILE'],inputs['DEV_TAMIL'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-tamil")

        return {"english_encoded_file":english_encoded_file,"tamil_encoded_file":tamil_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "tamil_dev_encoded_file":tamil_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}  

    except Exception as e:
        logger.error("error in english_tamil pairwise preprocessing: {}".format(e))
   
def english_and_hindi(inputs):
    "Exp-5.10: over 5.6 exp ..BPE 24k each, nolowercasing,pretok,shuffling"
    "steps:1.tokenize hindi using indicnlp, english using moses"
    "      2.train sp models for hindi and english and then encode train, dev, test files "
    "      3.preprocess nmt and embeddings"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_hindi = 'hi_{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'en_{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_hindi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_hindi','model_en-hi_{}_{}-model'.format(experiment_key,date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_hindi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_hindi'))
            logger.info("folder created at {}".format(model_intermediate_folder))
        hindi_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_train_tok'+unique_id+'.txt')
        hindi_dev_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_dev_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        english_test_Gen_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_Gen_tok.txt')
        english_test_LC_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_LC_tok.txt')
        hindi_test_Gen_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_test_Gen_tok.txt')
        hindi_test_LC_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_test_LC_tok.txt')
        hindi_encoded_file = os.path.join(model_master_train_folder, 'hindi_train_final'+unique_id+'.txt')
        hindi_dev_encoded_file = os.path.join(model_master_train_folder, 'hindi_dev_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        english_test_Gen_encoded_file = os.path.join(model_master_train_folder, 'english_test_Gen_final'+unique_id+'.txt')
        english_test_LC_encoded_file = os.path.join(model_master_train_folder, 'english_test_LC_final'+unique_id+'.txt')
        hindi_test_Gen_encoded_file = os.path.join(model_master_train_folder, 'hindi_test_Gen_final'+unique_id+'.txt')
        hindi_test_LC_encoded_file = os.path.join(model_master_train_folder, 'hindi_test_LC_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-hin pairwise preprocessing, startting for exp:{}".format(experiment_key))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(inputs['HINDI_TRAIN_FILE'], hindi_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(inputs['DEV_HINDI'], hindi_dev_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['TEST_HINDI_GEN'], hindi_test_Gen_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['TEST_HINDI_LC'], hindi_test_LC_tokenized_file))
        logger.info("Eng-hin pairwise preprocessing, hindi train,dev,test corpus tokenized")
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_GEN'], english_test_Gen_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_LC'], english_test_LC_tokenized_file))
        logger.info("Eng-hin pairwise preprocessing, english train,dev,test corpus tokenized")
        sp.train_spm(hindi_tokenized_file,sp_model_prefix_hindi, 24000, 'bpe')
        logger.info("Eng-hin pairwise preprocessing,sentencepiece model hindi trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("Eng-hin pairwise preprocessing,sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_tokenized_file,hindi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_dev_tokenized_file,hindi_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_test_Gen_tokenized_file,hindi_test_Gen_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_test_LC_tokenized_file,hindi_test_LC_encoded_file)
        logger.info("hindi-train,dev,test encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_Gen_tokenized_file,english_test_Gen_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_LC_tokenized_file,english_test_LC_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(hindi_tokenized_file,hindi_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['HINDI_TRAIN_FILE'],inputs['DEV_HINDI'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-hindi")

        os.system('rm- f {0} {1} {2} {3}'.format(english_test_Gen_encoded_file,english_test_LC_encoded_file,hindi_test_Gen_encoded_file,hindi_test_LC_encoded_file))

        return {"english_encoded_file":english_encoded_file,"hindi_encoded_file":hindi_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "hindi_dev_encoded_file":hindi_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}


    except Exception as e:
        print(e)
        logger.info("error in english_hindi anuvaad script: {}".format(e))


def english_and_marathi(inputs):
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_marathi = 'marathi-{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'enMr-{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_marathi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_marathi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_marathi','model_enMr-{}_{}-model'.format(experiment_key,date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_marathi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_marathi'))
            logger.info("folder created at {}".format(model_intermediate_folder))
        
        marathi_tokenized_file = os.path.join(model_intermediate_folder, 'marathi_train_tok'+unique_id+'.txt')
        marathi_dev_tokenized_file = os.path.join(model_intermediate_folder, 'marathi_dev_tok'+unique_id+'.txt')
        marathi_test_tokenized_file = os.path.join(model_intermediate_folder, 'marathi_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        marathi_encoded_file = os.path.join(model_master_train_folder, 'marathi_train_final'+unique_id+'.txt')
        marathi_dev_encoded_file = os.path.join(model_master_train_folder, 'marathi_dev_final'+unique_id+'.txt')
        marathi_test_encoded_file = os.path.join(model_master_train_folder, 'marathi_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-marathi pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} mr'.format(inputs['MARATHI_TRAIN_FILE'], marathi_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} mr'.format(inputs['DEV_MARATHI'], marathi_dev_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} mr'.format(mcl.english_marathi['TEST_MARATHI'], marathi_test_tokenized_file))
        logger.info("Eng-marathi pairwise preprocessing, marathi train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_marathi['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-marathi pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(marathi_tokenized_file,sp_model_prefix_marathi, 24000, 'bpe')
        logger.info("sentencepiece model marathi trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_marathi+'.model')),marathi_tokenized_file,marathi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_marathi+'.model')),marathi_dev_tokenized_file,marathi_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_marathi+'.model')),marathi_test_tokenized_file,marathi_test_encoded_file)
        logger.info("marathi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(marathi_tokenized_file,marathi_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['MARATHI_TRAIN_FILE'],inputs['DEV_MARATHI'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-marathi")           

        return {"english_encoded_file":english_encoded_file,"marathi_encoded_file":marathi_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "marathi_dev_encoded_file":marathi_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}  

    except Exception as e:
        logger.error("error in english_marathi pairwise preprocessing: {}".format(e))        