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
TRAIN_LOG_FILE = 'intermediate_data/anuvaad_training_log_file.txt'

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
        # tamil_test_tokenized_file = os.path.join(model_intermediate_folder, 'tamil_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        # english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        tamil_encoded_file = os.path.join(model_master_train_folder, 'tamil_train_final'+unique_id+'.txt')
        tamil_dev_encoded_file = os.path.join(model_master_train_folder, 'tamil_dev_final'+unique_id+'.txt')
        # tamil_test_encoded_file = os.path.join(model_master_train_folder, 'tamil_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        # english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-tamil pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} ta'.format(inputs['TAMIL_TRAIN_FILE'], tamil_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} ta'.format(inputs['DEV_TAMIL'], tamil_dev_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} ta'.format(mcl.english_tamil['TEST_TAMIL'], tamil_test_tokenized_file))
        logger.info("Eng-tamil pairwise preprocessing, tamil train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_tamil['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-tamil pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(tamil_tokenized_file,sp_model_prefix_tamil, 24000, 'bpe')
        logger.info("sentencepiece model tamil trained at {}".format(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model'))))
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained at {}".format(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model'))))
        
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),tamil_tokenized_file,tamil_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),tamil_dev_tokenized_file,tamil_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),tamil_test_tokenized_file,tamil_test_encoded_file)
        logger.info("tamil-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
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


def english_and_gujarati(inputs):
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_gujarati = 'gujarati-{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'enGuj-{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_gujarati')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_gujarati')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_gujarati','model_enGuj-{}_{}-model'.format(experiment_key,date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_gujarati'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_gujarati'))
            logger.info("folder created at {}".format(model_intermediate_folder))
        
        gujarati_tokenized_file = os.path.join(model_intermediate_folder, 'gujarati_train_tok'+unique_id+'.txt')
        gujarati_dev_tokenized_file = os.path.join(model_intermediate_folder, 'gujarati_dev_tok'+unique_id+'.txt')
        gujarati_test_tokenized_file = os.path.join(model_intermediate_folder, 'gujarati_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        gujarati_encoded_file = os.path.join(model_master_train_folder, 'gujarati_train_final'+unique_id+'.txt')
        gujarati_dev_encoded_file = os.path.join(model_master_train_folder, 'gujarati_dev_final'+unique_id+'.txt')
        gujarati_test_encoded_file = os.path.join(model_master_train_folder, 'gujarati_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-gujarati pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} gu'.format(inputs['GUJARATI_TRAIN_FILE'], gujarati_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} gu'.format(inputs['DEV_GUJARATI'], gujarati_dev_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} gu'.format(mcl.english_gujarati['TEST_GUJARATI'], gujarati_test_tokenized_file))
        logger.info("Eng-gujarati pairwise preprocessing, gujarati train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_gujarati['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-gujarati pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(gujarati_tokenized_file,sp_model_prefix_gujarati, 24000, 'bpe')
        logger.info("sentencepiece model gujarati trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_gujarati+'.model')),gujarati_tokenized_file,gujarati_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_gujarati+'.model')),gujarati_dev_tokenized_file,gujarati_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_gujarati+'.model')),gujarati_test_tokenized_file,gujarati_test_encoded_file)
        logger.info("gujarati-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(gujarati_tokenized_file,gujarati_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['GUJARATI_TRAIN_FILE'],inputs['DEV_GUJARATI'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-gujarati")           

        return {"english_encoded_file":english_encoded_file,"gujarati_encoded_file":gujarati_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "gujarati_dev_encoded_file":gujarati_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}  

    except Exception as e:
        logger.error("error in english_gujarati pairwise preprocessing: {}".format(e))        

def english_and_bengali(inputs):
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_bengali = 'bengali-{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'enBeng-{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_bengali')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_bengali')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_bengali','model_enBeng-{}_{}-model'.format(experiment_key,date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_bengali'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_bengali'))
            logger.info("folder created at {}".format(model_intermediate_folder))
        
        bengali_tokenized_file = os.path.join(model_intermediate_folder, 'bengali_train_tok'+unique_id+'.txt')
        bengali_dev_tokenized_file = os.path.join(model_intermediate_folder, 'bengali_dev_tok'+unique_id+'.txt')
        bengali_test_tokenized_file = os.path.join(model_intermediate_folder, 'bengali_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        bengali_encoded_file = os.path.join(model_master_train_folder, 'bengali_train_final'+unique_id+'.txt')
        bengali_dev_encoded_file = os.path.join(model_master_train_folder, 'bengali_dev_final'+unique_id+'.txt')
        bengali_test_encoded_file = os.path.join(model_master_train_folder, 'bengali_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-bengali pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} bn'.format(inputs['BENGALI_TRAIN_FILE'], bengali_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} bn'.format(inputs['DEV_BENGALI'], bengali_dev_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} bn'.format(mcl.english_bengali['TEST_BENGALI'], bengali_test_tokenized_file))
        logger.info("Eng-bengali pairwise preprocessing, bengali train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_bengali['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-bengali pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(bengali_tokenized_file,sp_model_prefix_bengali, 24000, 'bpe')
        logger.info("sentencepiece model bengali trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_bengali+'.model')),bengali_tokenized_file,bengali_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_bengali+'.model')),bengali_dev_tokenized_file,bengali_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_bengali+'.model')),bengali_test_tokenized_file,bengali_test_encoded_file)
        logger.info("bengali-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(bengali_tokenized_file,bengali_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['BENGALI_TRAIN_FILE'],inputs['DEV_BENGALI'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-bengali")           

        return {"english_encoded_file":english_encoded_file,"bengali_encoded_file":bengali_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "bengali_dev_encoded_file":bengali_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}  

    except Exception as e:
        logger.error("error in english_bengali pairwise preprocessing: {}".format(e))

def english_and_kannada(inputs):
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_kannada = 'kannada-{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'enKann-{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_kannada')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_kannada')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_kannada','model_enKann-{}_{}-model'.format(experiment_key,date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_kannada'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_kannada'))
            logger.info("folder created at {}".format(model_intermediate_folder))
        
        kannada_tokenized_file = os.path.join(model_intermediate_folder, 'kannada_train_tok'+unique_id+'.txt')
        kannada_dev_tokenized_file = os.path.join(model_intermediate_folder, 'kannada_dev_tok'+unique_id+'.txt')
        # kannada_test_tokenized_file = os.path.join(model_intermediate_folder, 'kannada_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        # english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        kannada_encoded_file = os.path.join(model_master_train_folder, 'kannada_train_final'+unique_id+'.txt')
        kannada_dev_encoded_file = os.path.join(model_master_train_folder, 'kannada_dev_final'+unique_id+'.txt')
        # kannada_test_encoded_file = os.path.join(model_master_train_folder, 'kannada_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        # english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-kannada pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} kn'.format(inputs['KANNADA_TRAIN_FILE'], kannada_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} kn'.format(inputs['DEV_KANNADA'], kannada_dev_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} kn'.format(mcl.english_kannada['TEST_KANNADA'], kannada_test_tokenized_file))
        logger.info("Eng-kannada pairwise preprocessing, kannada train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_kannada['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-kannada pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(kannada_tokenized_file,sp_model_prefix_kannada, 24000, 'bpe')
        logger.info("sentencepiece model kannada trained at {}".format(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_kannada+'.model'))))
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained at {}".format(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model'))))

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_kannada+'.model')),kannada_tokenized_file,kannada_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_kannada+'.model')),kannada_dev_tokenized_file,kannada_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_kannada+'.model')),kannada_test_tokenized_file,kannada_test_encoded_file)
        logger.info("kannada-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(kannada_tokenized_file,kannada_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['KANNADA_TRAIN_FILE'],inputs['DEV_KANNADA'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-kannada")           

        return {"english_encoded_file":english_encoded_file,"kannada_encoded_file":kannada_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "kannada_dev_encoded_file":kannada_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}  

    except Exception as e:
        logger.error("error in english_kannada pairwise preprocessing: {}".format(e))                


def english_and_telugu(inputs):
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_telugu = 'telugu-{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'enTelg-{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_telugu')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_telugu')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_telugu','model_enTelg-{}_{}-model'.format(experiment_key,date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_telugu'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_telugu'))
            logger.info("folder created at {}".format(model_intermediate_folder))
        
        telugu_tokenized_file = os.path.join(model_intermediate_folder, 'telugu_train_tok'+unique_id+'.txt')
        telugu_dev_tokenized_file = os.path.join(model_intermediate_folder, 'telugu_dev_tok'+unique_id+'.txt')
        telugu_test_tokenized_file = os.path.join(model_intermediate_folder, 'telugu_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        telugu_encoded_file = os.path.join(model_master_train_folder, 'telugu_train_final'+unique_id+'.txt')
        telugu_dev_encoded_file = os.path.join(model_master_train_folder, 'telugu_dev_final'+unique_id+'.txt')
        telugu_test_encoded_file = os.path.join(model_master_train_folder, 'telugu_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-telugu pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} te'.format(inputs['TELUGU_TRAIN_FILE'], telugu_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} te'.format(inputs['DEV_TELUGU'], telugu_dev_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} te'.format(mcl.english_telugu['TEST_TELUGU'], telugu_test_tokenized_file))
        logger.info("Eng-telugu pairwise preprocessing, telugu train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_telugu['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-telugu pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(telugu_tokenized_file,sp_model_prefix_telugu, 24000, 'bpe')
        logger.info("sentencepiece model telugu trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_telugu+'.model')),telugu_tokenized_file,telugu_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_telugu+'.model')),telugu_dev_tokenized_file,telugu_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_telugu+'.model')),telugu_test_tokenized_file,telugu_test_encoded_file)
        logger.info("telugu-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(telugu_tokenized_file,telugu_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['TELUGU_TRAIN_FILE'],inputs['DEV_TELUGU'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-telugu")           

        return {"english_encoded_file":english_encoded_file,"telugu_encoded_file":telugu_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "telugu_dev_encoded_file":telugu_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}  

    except Exception as e:
        logger.error("error in english_telugu pairwise preprocessing: {}".format(e))

def english_and_malayalam(inputs):
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_malayalam = 'malayalam-{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'enMalay-{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_malayalam')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_malayalam')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_malayalam','model_enMalay-{}_{}-model'.format(experiment_key,date_now))
        
        paths = [model_intermediate_folder,model_master_train_folder,os.path.join(NMT_MODEL_DIR, 'english_malayalam')]
        if_path_exists = [os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_malayalam'))]
        indices = [i for i, val in enumerate(if_path_exists) if not val]
        [os.makedirs(paths[i]) for i in indices if len(indices)>0]
        logger.info("folder created ")

        # if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_malayalam'))]):
        #     print("inside folder creation")
        #     os.makedirs(model_intermediate_folder)
        #     os.makedirs(model_master_train_folder)
        #     os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_malayalam'))
        #     logger.info("folder created at {}".format(model_intermediate_folder))
        
        malayalam_tokenized_file = os.path.join(model_intermediate_folder, 'malayalam_train_tok'+unique_id+'.txt')
        malayalam_dev_tokenized_file = os.path.join(model_intermediate_folder, 'malayalam_dev_tok'+unique_id+'.txt')
        malayalam_test_tokenized_file = os.path.join(model_intermediate_folder, 'malayalam_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        malayalam_encoded_file = os.path.join(model_master_train_folder, 'malayalam_train_final'+unique_id+'.txt')
        malayalam_dev_encoded_file = os.path.join(model_master_train_folder, 'malayalam_dev_final'+unique_id+'.txt')
        malayalam_test_encoded_file = os.path.join(model_master_train_folder, 'malayalam_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-malayalam pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} ml'.format(inputs['MALAYALAM_TRAIN_FILE'], malayalam_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} ml'.format(inputs['DEV_MALAYALAM'], malayalam_dev_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} ml'.format(mcl.english_malayalam['TEST_MALAYALAM'], malayalam_test_tokenized_file))
        logger.info("Eng-malayalam pairwise preprocessing, malayalam train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_malayalam['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-malayalam pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(malayalam_tokenized_file,sp_model_prefix_malayalam, 24000, 'bpe')
        logger.info("sentencepiece model malayalam trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_malayalam+'.model')),malayalam_tokenized_file,malayalam_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_malayalam+'.model')),malayalam_dev_tokenized_file,malayalam_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_malayalam+'.model')),malayalam_test_tokenized_file,malayalam_test_encoded_file)
        logger.info("malayalam-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(malayalam_tokenized_file,malayalam_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['MALAYALAM_TRAIN_FILE'],inputs['DEV_MALAYALAM'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-malayalam")           

        return {"english_encoded_file":english_encoded_file,"malayalam_encoded_file":malayalam_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "malayalam_dev_encoded_file":malayalam_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}  

    except Exception as e:
        logger.error("error in english_malayalam pairwise preprocessing: {}".format(e))  


def english_and_punjabi(inputs):
    try:
        experiment_key = inputs['experiment_key']
        unique_id = inputs['unique_id']
        sp_model_prefix_punjabi = 'punjabi-{}-{}-24k'.format(experiment_key,date_now)
        sp_model_prefix_english = 'enPun-{}-{}-24k'.format(experiment_key,date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_punjabi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_punjabi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_punjabi','model_enPun-{}_{}-model'.format(experiment_key,date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_punjabi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_punjabi'))
            logger.info("folder created at {}".format(model_intermediate_folder))
        
        punjabi_tokenized_file = os.path.join(model_intermediate_folder, 'punjabi_train_tok'+unique_id+'.txt')
        punjabi_dev_tokenized_file = os.path.join(model_intermediate_folder, 'punjabi_dev_tok'+unique_id+'.txt')
        punjabi_test_tokenized_file = os.path.join(model_intermediate_folder, 'punjabi_test_tok'+unique_id+'.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok'+unique_id+'.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok'+unique_id+'.txt')
        english_test_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_tok'+unique_id+'.txt')
        punjabi_encoded_file = os.path.join(model_master_train_folder, 'punjabi_train_final'+unique_id+'.txt')
        punjabi_dev_encoded_file = os.path.join(model_master_train_folder, 'punjabi_dev_final'+unique_id+'.txt')
        punjabi_test_encoded_file = os.path.join(model_master_train_folder, 'punjabi_test_final'+unique_id+'.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final'+unique_id+'.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final'+unique_id+'.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final'+unique_id+'.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data-{}_{}'.format(experiment_key,date_now))

        logger.info("Eng-punjabi pairwise preprocessing, startting for exp:{}".format(experiment_key))

        os.system('python ./tools/indic_tokenize.py {0} {1} pa'.format(inputs['PUNJABI_TRAIN_FILE'], punjabi_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} pa'.format(inputs['DEV_PUNJABI'], punjabi_dev_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} pa'.format(mcl.english_punjabi['TEST_PUNJABI'], punjabi_test_tokenized_file))
        logger.info("Eng-punjabi pairwise preprocessing, punjabi train,dev,test corpus tokenized")

        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(inputs['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_punjabi['TEST_ENGLISH'], english_test_tokenized_file))
        logger.info("Eng-punjabi pairwise preprocessing, english train,dev,test corpus tokenized")

        sp.train_spm(punjabi_tokenized_file,sp_model_prefix_punjabi, 24000, 'bpe')
        logger.info("sentencepiece model punjabi trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_punjabi+'.model')),punjabi_tokenized_file,punjabi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_punjabi+'.model')),punjabi_dev_tokenized_file,punjabi_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_punjabi+'.model')),punjabi_test_tokenized_file,punjabi_test_encoded_file)
        logger.info("punjabi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_tokenized_file,english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7}'.format(punjabi_tokenized_file,punjabi_dev_tokenized_file,english_tokenized_file,english_dev_tokenized_file,\
                   inputs['PUNJABI_TRAIN_FILE'],inputs['DEV_PUNJABI'],inputs['ENGLISH_TRAIN_FILE'],inputs['DEV_ENGLISH']))
        logger.info("removed intermediate files: pairwise preporcessing: eng-punjabi")           

        return {"english_encoded_file":english_encoded_file,"punjabi_encoded_file":punjabi_encoded_file,"english_dev_encoded_file":english_dev_encoded_file, \
               "punjabi_dev_encoded_file":punjabi_dev_encoded_file,"nmt_processed_data":nmt_processed_data,"nmt_model_path":nmt_model_path}  

    except Exception as e:
        logger.error("error in english_punjabi pairwise preprocessing: {}".format(e))                      