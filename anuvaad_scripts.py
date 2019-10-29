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

def english_hindi():
    "in progress"
    "steps:1.tokenize hindi using indicnlp, english using moses"
    "      2.train sp models for hindi and english and then encode train, dev, test files "
    "      3.preprocess nmt and embeddings"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_hindi = 'hi-{}-10k'.format(date_now)
        sp_model_prefix_english = 'en-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_hindi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_hindi','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_hindi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_hindi'))
            print("folder created at {}".format(model_intermediate_folder))
        hindi_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_train_tok.txt')
        hindi_dev_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_dev_tok.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok.txt')
        english_test_Gen_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_Gen_tok.txt')
        english_test_LC_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_LC_tok.txt')
        english_test_TB_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_TB_tok.txt')
        hindi_encoded_file = os.path.join(model_master_train_folder, 'hindi_train_final.txt')
        hindi_dev_encoded_file = os.path.join(model_master_train_folder, 'hindi_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_Gen_encoded_file = os.path.join(model_master_train_folder, 'english_test_Gen_final.txt')
        english_test_LC_encoded_file = os.path.join(model_master_train_folder, 'english_test_LC_final.txt')
        english_test_TB_encoded_file = os.path.join(model_master_train_folder, 'english_test_TB_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        # os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['HINDI_TRAIN_FILE'], hindi_tokenized_file))
        # os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['DEV_HINDI'], hindi_dev_tokenized_file))
        # logger.info("english-hindi, hindi train and dev corpus tokenized")
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_GEN'], english_test_Gen_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_LC'], english_test_LC_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_TB'], english_test_TB_tokenized_file))
        # logger.info("english-hindi, english train, dev,test corpus tokenized")
        sp.train_spm(mcl.english_hindi['HINDI_TRAIN_FILE'],sp_model_prefix_hindi, 10000, 'bpe')
        logger.info("sentencepiece model hindi trained")
        sp.train_spm(mcl.english_hindi['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),mcl.english_hindi['HINDI_TRAIN_FILE'],hindi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),mcl.english_hindi['DEV_HINDI'],hindi_dev_encoded_file)
        logger.info("hindi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_hindi['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_hindi['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_hindi['TEST_ENGLISH_GEN'],english_test_Gen_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_hindi['TEST_ENGLISH_LC'],english_test_LC_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_hindi['TEST_ENGLISH_TB'],english_test_TB_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 150 -tgt_seq_length 150 -save_data {4}'.format(english_encoded_file,hindi_encoded_file,english_dev_encoded_file,hindi_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('python ./embeddings_to_torch.py -emb_file_enc ~/glove/glove.6B.300d.txt -emb_file_dec ~/glove/cc.hi.300.vec -dict_file {0} -output_file {1}'.format(nmt_processed_data+'.vocab.pt',model_master_train_folder+'embeddings_eng_hin'))
        print("glove embedding done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -pre_word_vecs_enc {2} -pre_word_vecs_dec {3} -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 6000 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0'.format(nmt_processed_data,nmt_model_path,model_master_train_folder+'embeddings_eng_hin.enc.pt',model_master_train_folder+'embeddings_eng_hin.dec.pt'))

    except Exception as e:
        print(e)
        logger.info("error in english_hindi anuvaad script: {}".format(e))

def english_hindi_experiments():

    "29/10/19: Exp-12: old_data_original+lc_cleaned+ ik names translated from google(100k)+shabdkosh(appended 29k new),BPE-24K,50knmt"
    "steps:1.tokenize hindi using indicnlp, english using moses"
    "      2.train sp models for hindi and english and then encode train, dev, test files "
    "      3.preprocess nmt and embeddings"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_hindi = 'hi_exp-12-{}-24k'.format(date_now)
        sp_model_prefix_english = 'en_exp-12-{}-24k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_hindi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_hindi','model_en-hi_exp-12_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_hindi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_hindi'))
            print("folder created at {}".format(model_intermediate_folder))
        hindi_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_train_tok.txt')
        hindi_dev_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_dev_tok.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok.txt')
        english_test_Gen_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_Gen_tok.txt')
        english_test_LC_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_LC_tok.txt')
        english_test_TB_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_TB_tok.txt')
        hindi_encoded_file = os.path.join(model_master_train_folder, 'hindi_train_final.txt')
        hindi_dev_encoded_file = os.path.join(model_master_train_folder, 'hindi_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_Gen_encoded_file = os.path.join(model_master_train_folder, 'english_test_Gen_final.txt')
        english_test_LC_encoded_file = os.path.join(model_master_train_folder, 'english_test_LC_final.txt')
        english_test_TB_encoded_file = os.path.join(model_master_train_folder, 'english_test_TB_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_exp-12_{}'.format(date_now))

        print("Exp-12 training")
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['HINDI_TRAIN_FILE'], hindi_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['DEV_HINDI'], hindi_dev_tokenized_file))
        logger.info("english-hindi, hindi train and dev corpus tokenized")
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['DEV_ENGLISH'], english_dev_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_GEN'], english_test_Gen_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_LC'], english_test_LC_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_TB'], english_test_TB_tokenized_file))
        logger.info("english-hindi, english train, dev,test corpus tokenized")
        sp.train_spm(hindi_tokenized_file,sp_model_prefix_hindi, 24000, 'bpe')
        logger.info("sentencepiece model hindi trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_tokenized_file,hindi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_dev_tokenized_file,hindi_dev_encoded_file)
        logger.info("hindi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_Gen_tokenized_file,english_test_Gen_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_LC_tokenized_file,english_test_LC_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_TB_tokenized_file,english_test_TB_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 150 -tgt_seq_length 150 -save_data {4}'.format(english_encoded_file,hindi_encoded_file,english_dev_encoded_file,hindi_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('python ./embeddings_to_torch.py -emb_file_enc ~/glove/glove.6B.300d.txt -emb_file_dec ~/glove/cc.hi.300.vec -dict_file {0} -output_file {1}'.format(nmt_processed_data+'.vocab.pt',os.path.join(model_master_train_folder,'embeddings_eng_hin')))
        print("glove embedding done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -pre_word_vecs_enc {2} -pre_word_vecs_dec {3} -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 150000  -max_generator_batches 2 -dropout 0.1 -batch_size 6000 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0'.format(nmt_processed_data,nmt_model_path,os.path.join(model_master_train_folder,'embeddings_eng_hin.enc.pt'),os.path.join(model_master_train_folder,'embeddings_eng_hin.dec.pt')))

    except Exception as e:
        print(e)
        logger.info("error in english_hindi anuvaad script: {}".format(e))

def english_hindi_experiments_word_based():

    "exp-4.Word based model + tokenization +6000 on 1gpu +all lowercasing"
    "steps:1.tokenize hindi using indicnlp, english using moses"
    "      3.preprocess nmt and embeddings"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: similarly nmt model and preprocess.py"
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_hindi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_hindi','model_en-hi_exp-4_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_hindi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_hindi'))
            print("folder created at {}".format(model_intermediate_folder))
        hindi_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_train_tok.txt')
        hindi_dev_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_dev_tok.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok.txt')
        english_test_Gen_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_Gen_tok.txt')
        english_test_LC_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_LC_tok.txt')
        english_test_TB_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_TB_tok.txt')
        hindi_encoded_file = os.path.join(model_master_train_folder, 'hindi_train_final.txt')
        hindi_dev_encoded_file = os.path.join(model_master_train_folder, 'hindi_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_Gen_encoded_file = os.path.join(model_master_train_folder, 'english_test_Gen_final.txt')
        english_test_LC_encoded_file = os.path.join(model_master_train_folder, 'english_test_LC_final.txt')
        english_test_TB_encoded_file = os.path.join(model_master_train_folder, 'english_test_TB_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_exp-4_{}'.format(date_now))

        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['HINDI_TRAIN_FILE'], hindi_encoded_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['DEV_HINDI'], hindi_dev_encoded_file))
        logger.info("english-hindi, hindi train and dev corpus tokenized")
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['ENGLISH_TRAIN_FILE'], english_encoded_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['DEV_ENGLISH'], english_dev_encoded_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_GEN'], english_test_Gen_encoded_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_LC'], english_test_LC_encoded_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_TB'], english_test_TB_encoded_file))
        logger.info("english-hindi, english train, dev,test corpus tokenized")
        print("english-hindi, english train, dev,test corpus tokenized")
        # sp.train_spm(hindi_tokenized_file,sp_model_prefix_hindi, 10000, 'unigram')
        # logger.info("sentencepiece model hindi trained")
        # sp.train_spm(english_tokenized_file,sp_model_prefix_english, 10000, 'unigram')
        # logger.info("sentencepiece model english trained")

        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_tokenized_file,hindi_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_dev_tokenized_file,hindi_dev_encoded_file)
        # logger.info("hindi-train file and dev encoded and final stored in data folder")
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_Gen_tokenized_file,english_test_Gen_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_LC_tokenized_file,english_test_LC_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_TB_tokenized_file,english_test_TB_encoded_file)
        # logger.info("english-train,dev,test file encoded and final stored in data folder")
        # print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 150 -tgt_seq_length 150 -save_data {4}'.format(english_encoded_file,hindi_encoded_file,english_dev_encoded_file,hindi_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('python ./embeddings_to_torch.py -emb_file_enc ~/glove/glove.6B.300d.txt -emb_file_dec ~/glove/cc.hi.300.vec -dict_file {0} -output_file {1}'.format(nmt_processed_data+'.vocab.pt',os.path.join(model_master_train_folder,'embeddings_eng_hin')))
        print("glove embedding done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -pre_word_vecs_enc {2} -pre_word_vecs_dec {3} -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 140000  -max_generator_batches 2 -dropout 0.1 -batch_size 6000 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0'.format(nmt_processed_data,nmt_model_path,os.path.join(model_master_train_folder,'embeddings_eng_hin.enc.pt'),os.path.join(model_master_train_folder,'embeddings_eng_hin.dec.pt')))

    except Exception as e:
        print(e)
        logger.info("error in english_hindi anuvaad script: {}".format(e))

def english_tamil():
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

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200 -save_data {4}'.format(english_encoded_file,tamil_encoded_file,english_dev_encoded_file,tamil_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 2 -gpu_ranks 0 1'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_tamil anuvaad script: {}".format(e))

def english_gujrati():
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for gujrati and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_gujrati = 'guj-{}-10k'.format(date_now)
        sp_model_prefix_english = 'en-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_gujrati')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_gujrati')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_gujrati','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_gujrati'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_gujrati'))
            print("folder created at {}".format(model_intermediate_folder))
        gujrati_encoded_file = os.path.join(model_master_train_folder, 'gujrati_train_final.txt')
        gujrati_dev_encoded_file = os.path.join(model_master_train_folder, 'gujrati_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        sp.train_spm(mcl.english_gujrati['GUJRATI_TRAIN_FILE'],sp_model_prefix_gujrati, 10000, 'bpe')
        logger.info("sentencepiece model gujrati trained")
        sp.train_spm(mcl.english_gujrati['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_gujrati+'.model')),mcl.english_gujrati['GUJRATI_TRAIN_FILE'],gujrati_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_gujrati+'.model')),mcl.english_gujrati['DEV_GUJRATI'],gujrati_dev_encoded_file)
        logger.info("gujrati-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_gujrati['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_gujrati['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_gujrati['TEST_ENGLISH'],english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 150 -tgt_seq_length 150 -save_data {4}'.format(english_encoded_file,gujrati_encoded_file,english_dev_encoded_file,gujrati_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 2 -gpu_ranks 0 1'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_gujrati anuvaad script: {}".format(e))

def english_bengali(): 
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for bengali and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_bengali = 'beng-{}-10k'.format(date_now)
        sp_model_prefix_english = 'en-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_bengali')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_bengali')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_bengali','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_bengali'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_bengali'))
            print("folder created at {}".format(model_intermediate_folder))
        bengali_encoded_file = os.path.join(model_master_train_folder, 'bengali_train_final.txt')
        bengali_dev_encoded_file = os.path.join(model_master_train_folder, 'bengali_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        sp.train_spm(mcl.english_bengali['BENGALI_TRAIN_FILE'],sp_model_prefix_bengali, 10000, 'bpe')
        logger.info("sentencepiece model bengali trained")
        sp.train_spm(mcl.english_bengali['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_bengali+'.model')),mcl.english_bengali['BENGALI_TRAIN_FILE'],bengali_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_bengali+'.model')),mcl.english_bengali['DEV_BENGALI'],bengali_dev_encoded_file)
        logger.info("bengali-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_bengali['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_bengali['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_bengali['TEST_ENGLISH'],english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 150 -tgt_seq_length 150 -save_data {4}'.format(english_encoded_file,bengali_encoded_file,english_dev_encoded_file,bengali_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_bengali anuvaad script: {}".format(e))   

def english_marathi(): 
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for marathi and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_marathi = 'marathi-{}-10k'.format(date_now)
        sp_model_prefix_english = 'enMr-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_marathi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_marathi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_marathi','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_marathi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_marathi'))
            print("folder created at {}".format(model_intermediate_folder))
        marathi_encoded_file = os.path.join(model_master_train_folder, 'marathi_train_final.txt')
        marathi_dev_encoded_file = os.path.join(model_master_train_folder, 'marathi_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        sp.train_spm(mcl.english_marathi['MARATHI_TRAIN_FILE'],sp_model_prefix_marathi, 10000, 'bpe')
        logger.info("sentencepiece model marathi trained")
        sp.train_spm(mcl.english_marathi['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_marathi+'.model')),mcl.english_marathi['MARATHI_TRAIN_FILE'],marathi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_marathi+'.model')),mcl.english_marathi['DEV_MARATHI'],marathi_dev_encoded_file)
        logger.info("marathi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_marathi['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_marathi['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_marathi['TEST_ENGLISH'],english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200 -save_data {4}'.format(english_encoded_file,marathi_encoded_file,english_dev_encoded_file,marathi_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 2 -gpu_ranks 0 1'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_marathi anuvaad script: {}".format(e)) 

def english_kannada(): 
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for kannada and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_kannada = 'kannada-{}-10k'.format(date_now)
        sp_model_prefix_english = 'enKn-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_kannada')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_kannada')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_kannada','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_kannada'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_kannada'))
            print("folder created at {}".format(model_intermediate_folder))
        kannada_encoded_file = os.path.join(model_master_train_folder, 'kannada_train_final.txt')
        kannada_dev_encoded_file = os.path.join(model_master_train_folder, 'kannada_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        sp.train_spm(mcl.english_kannada['KANNADA_TRAIN_FILE'],sp_model_prefix_kannada, 10000, 'bpe')
        logger.info("sentencepiece model kannada trained")
        sp.train_spm(mcl.english_kannada['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_kannada+'.model')),mcl.english_kannada['KANNADA_TRAIN_FILE'],kannada_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_kannada+'.model')),mcl.english_kannada['DEV_KANNADA'],kannada_dev_encoded_file)
        logger.info("kannada-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_kannada['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_kannada['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_kannada['TEST_ENGLISH'],english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200 -save_data {4}'.format(english_encoded_file,kannada_encoded_file,english_dev_encoded_file,kannada_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 2 -gpu_ranks 0 1'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_kannada anuvaad script: {}".format(e))         

def english_telgu(): 
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for telgu and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_telgu = 'telgu-{}-10k'.format(date_now)
        sp_model_prefix_english = 'enTe-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_telgu')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_telgu')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_telgu','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_telgu'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_telgu'))
            print("folder created at {}".format(model_intermediate_folder))
        telgu_encoded_file = os.path.join(model_master_train_folder, 'telgu_train_final.txt')
        telgu_dev_encoded_file = os.path.join(model_master_train_folder, 'telgu_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        sp.train_spm(mcl.english_telgu['TELGU_TRAIN_FILE'],sp_model_prefix_telgu, 10000, 'bpe')
        logger.info("sentencepiece model telgu trained")
        sp.train_spm(mcl.english_telgu['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_telgu+'.model')),mcl.english_telgu['TELGU_TRAIN_FILE'],telgu_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_telgu+'.model')),mcl.english_telgu['DEV_TELGU'],telgu_dev_encoded_file)
        logger.info("telgu-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_telgu['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_telgu['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_telgu['TEST_ENGLISH'],english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200 -save_data {4}'.format(english_encoded_file,telgu_encoded_file,english_dev_encoded_file,telgu_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 2 -gpu_ranks 0 1'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_telgu anuvaad script: {}".format(e))

def english_malayalam(): 
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for malayalam and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_malayalam = 'malayalam-{}-10k'.format(date_now)
        sp_model_prefix_english = 'enMl-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_malayalam')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_malayalam')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_malayalam','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_malayalam'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_malayalam'))
            print("folder created at {}".format(model_intermediate_folder))
        malayalam_encoded_file = os.path.join(model_master_train_folder, 'malayalam_train_final.txt')
        malayalam_dev_encoded_file = os.path.join(model_master_train_folder, 'malayalam_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        sp.train_spm(mcl.english_malayalam['MALAYALAM_TRAIN_FILE'],sp_model_prefix_malayalam, 10000, 'bpe')
        logger.info("sentencepiece model malayalam trained")
        sp.train_spm(mcl.english_malayalam['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_malayalam+'.model')),mcl.english_malayalam['MALAYALAM_TRAIN_FILE'],malayalam_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_malayalam+'.model')),mcl.english_malayalam['DEV_MALAYALAM'],malayalam_dev_encoded_file)
        logger.info("malayalam-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_malayalam['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_malayalam['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_malayalam['TEST_ENGLISH'],english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200 -save_data {4}'.format(english_encoded_file,malayalam_encoded_file,english_dev_encoded_file,malayalam_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 4 -gpu_ranks 0 1 2 3'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_malayalam anuvaad script: {}".format(e))

def english_punjabi(): 
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for punjabi and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_punjabi = 'punjabi-{}-10k'.format(date_now)
        sp_model_prefix_english = 'enPu-{}-10k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_punjabi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_punjabi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_punjabi','model_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_punjabi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_punjabi'))
            print("folder created at {}".format(model_intermediate_folder))
        punjabi_encoded_file = os.path.join(model_master_train_folder, 'punjabi_train_final.txt')
        punjabi_dev_encoded_file = os.path.join(model_master_train_folder, 'punjabi_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        english_test_encoded_file = os.path.join(model_master_train_folder, 'english_test_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        sp.train_spm(mcl.english_punjabi['PUNJABI_TRAIN_FILE'],sp_model_prefix_punjabi, 10000, 'bpe')
        logger.info("sentencepiece model punjabi trained")
        sp.train_spm(mcl.english_punjabi['ENGLISH_TRAIN_FILE'],sp_model_prefix_english, 10000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_punjabi+'.model')),mcl.english_punjabi['PUNJABI_TRAIN_FILE'],punjabi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_punjabi+'.model')),mcl.english_punjabi['DEV_PUNJABI'],punjabi_dev_encoded_file)
        logger.info("punjabi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_punjabi['ENGLISH_TRAIN_FILE'],english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_punjabi['DEV_ENGLISH'],english_dev_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),mcl.english_punjabi['TEST_ENGLISH'],english_test_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200 -save_data {4}'.format(english_encoded_file,punjabi_encoded_file,english_dev_encoded_file,punjabi_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 100000  -max_generator_batches 2 -dropout 0.1 -batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 4 -gpu_ranks 0 1 2 3'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_punjabi anuvaad script: {}".format(e))

if __name__ == '__main__':
    if sys.argv[1] == "english-tamil":
        english_tamil()
    elif sys.argv[1] == "english-hindi":
        english_hindi()
    elif sys.argv[1] == "english-gujrati":
        english_gujrati()
    elif sys.argv[1] == "english-bengali":
        english_bengali()
    elif sys.argv[1] == "english-marathi":
        english_marathi()
    elif sys.argv[1] == "english-kannada":
        english_kannada() 
    elif sys.argv[1] == "english-telgu":
        english_telgu()
    elif sys.argv[1] == "english-malayalam":
        english_malayalam() 
    elif sys.argv[1] == "english-punjabi":
        english_punjabi() 
    elif sys.argv[1] == "english-hindi-exp":
        english_hindi_experiments()
    elif sys.argv[1] == "english-hindi-exp-wb":
        english_hindi_experiments_word_based()                                    
    else:
        print("invalid request", sys.argv)
