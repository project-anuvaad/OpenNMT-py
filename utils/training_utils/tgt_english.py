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


def hindi_english_experiments():

    "29/10/19: hindi_eng_exp-1 exp 10+++  Old data + dictionary(also 29k aug),BPE-24, nolowercasing,pretok,shuffling"
    "steps:1.tokenize hindi using indicnlp, english using moses"
    "      2.train sp models for hindi and english and then encode train, dev, test files "
    "      3.preprocess nmt and embeddings"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_hindi = 'hi_exp_h-1-{}-24k'.format(date_now)
        sp_model_prefix_english = 'en_exp_h-1-{}-24k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_hindi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_hindi','model_hi-en_exp_h-1_{}-model'.format(date_now))
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
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_exp_h-1_{}'.format(date_now))

        print("Exp-1 hindi-english-29/10/19 training")
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['HINDI_TRAIN_FILE'], hindi_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['DEV_HINDI'], hindi_dev_tokenized_file))
        logger.info("english-hindi, hindi train and dev corpus tokenized")
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['DEV_ENGLISH'], english_dev_tokenized_file))
        sp.train_spm(hindi_tokenized_file,sp_model_prefix_hindi, 24000, 'bpe')
        logger.info("sentencepiece model hindi trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_tokenized_file,hindi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_hindi+'.model')),hindi_dev_tokenized_file,hindi_dev_encoded_file)
        logger.info("hindi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("hindi-eng-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 150 -tgt_seq_length 150 -save_data {4}'.format(hindi_encoded_file,english_encoded_file,hindi_dev_encoded_file,english_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('python ./embeddings_to_torch.py -emb_file_enc ~/glove/cc.hi.300.vec -emb_file_dec ~/glove/glove.6B.300d.txt -dict_file {0} -output_file {1}'.format(nmt_processed_data+'.vocab.pt',os.path.join(model_master_train_folder,'embeddings_hin_eng')))
        print("glove embedding done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -pre_word_vecs_enc {2} -pre_word_vecs_dec {3} -transformer_ff 2048 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 150000  -max_generator_batches 2 -dropout 0.1 -batch_size 6000 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 4 -gpu_ranks 0 1 2 3'.format(nmt_processed_data,nmt_model_path,os.path.join(model_master_train_folder,'embeddings_hin_eng.enc.pt'),os.path.join(model_master_train_folder,'embeddings_hin_eng.dec.pt')))

    except Exception as e:
        print(e)
        logger.info("error in english_hindi anuvaad script: {}".format(e))


def tamil_to_english():
    "steps:1.not using tokenizer and external embedding"
    "      2.train sp models for tamil and english and then encode train, dev, test files "
    "      3.preprocess nmt"
    "      4.nmt-train, change hyperparamter manually, these are hardcoded for now"        
    "Note: SP model prefix is date wise, If training more than one DIFFERENT model in a single day, kindly keep this factor in mind and change prefix accordingly similarly nmt model and preprocess.py"
    try:
        sp_model_prefix_tamil = 'tamil-{}-24k'.format(date_now)
        sp_model_prefix_english = 'enTa-{}-24k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_tamil')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_tamil')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_tamil','model_ta-en_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_tamil'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_tamil'))
            print("folder created at {}".format(model_intermediate_folder))
        
        tamil_tokenized_file = os.path.join(model_intermediate_folder, 'tamil_train_tok.txt')
        tamil_dev_tokenized_file = os.path.join(model_intermediate_folder, 'tamil_dev_tok.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok.txt')
        tamil_encoded_file = os.path.join(model_master_train_folder, 'tamil_train_final.txt')
        tamil_dev_encoded_file = os.path.join(model_master_train_folder, 'tamil_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_{}'.format(date_now))

        print("tam-eng 1 09-01-20 training")
        os.system('python ./tools/indic_tokenize.py {0} {1} ta'.format(mcl.english_tamil['TAMIL_TRAIN_FILE'], tamil_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} ta'.format(mcl.english_tamil['DEV_TAMIL'], tamil_dev_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_tamil['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_tamil['DEV_ENGLISH'], english_dev_tokenized_file))

        sp.train_spm(tamil_tokenized_file,sp_model_prefix_tamil, 24000, 'bpe')
        logger.info("sentencepiece model tamil trained")
        sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),tamil_tokenized_file,tamil_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_tamil+'.model')),tamil_dev_tokenized_file,tamil_dev_encoded_file)
        logger.info("tamil-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200\
             -save_data {4}'.format(tamil_encoded_file,english_encoded_file,tamil_dev_encoded_file,english_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8\
              -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 150000  -max_generator_batches 2 -dropout 0.1\
                -batch_size 6000 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam\
                -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000\
                -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_tamil anuvaad script: {}".format(e))

if __name__ == '__main__':
    if sys.argv[1] == "tamil_to_english":
        tamil_to_english()
    elif sys.argv[1] == "hinidi-english-exp":
        hindi_english_experiments()                                             
    else:
        print("invalid request", sys.argv)
