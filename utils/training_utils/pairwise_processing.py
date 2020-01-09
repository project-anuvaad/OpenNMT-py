"Pairwise preprocessing aims to do task which are common across two language pair."
"That is it doesn't matter which way we are training, this preprocessing is one time and works for both"

"***************InProgress*************"

import sys
import os
from onmt.utils.logging import logger
import tools.indic_tokenize as hin_tokenizer
import tools.sp_enc_dec as sp
import datetime
import corpus.master_corpus_location as mcl

date_now = datetime.datetime.now().strftime('%Y-%m-%d')
INTERMEDIATE_DATA_LOCATION = 'intermediate_data/'
TRAIN_DEV_TEST_DATA_LOCATION = 'data/'
NMT_MODEL_DIR = 'model/'
SENTENCEPIECE_MODEL_DIR = 'model/sentencepiece_models/'


def english_and_tamil():
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for tamil and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_tamil = 'tamil-{}-10k'.format(date_now)
        sp_model_prefix_english = 'enTa-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_tamil')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_tamil')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_tamil','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_tamil'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_tamil'))
            print("folder created at {}".format(model_intermediate_folder))
        tamil_encoded_file = os.path.join(model_master_train_folder, 'tamil_train_final.txt')
        tamil_dev_encoded_file = os.path.join(model_master_train_folder, 'tamil_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        sp.train_spm(mcl.english_tamil['TAMIL_TRAIN_FILE'],sp_model_prefix_tamil, 10000, 'bpe')
        logger.info("sentencepiece model tamil trained")
        sp.train_spm(mcl.english_tamil['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),mcl.english_tamil['TAMIL_TRAIN_FILE'],tamil_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),mcl.english_tamil['DEV_TAMIL'],tamil_dev_encoded_file)
        logger.info("tamil-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_tamil['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_tamil['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_tamil['TEST_ENGLISH'],english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")


    except Exception as e:
        print(e)
        logger.info("error in english_tamil anuvaad script: {}".format(e))
        return False