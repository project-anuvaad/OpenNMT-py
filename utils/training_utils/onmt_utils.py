import sys
import os
from onmt.utils.logging import init_logger
import tools.indic_tokenize as hin_tokenizer
import tools.sp_enc_dec as sp
import datetime
import corpus.master_corpus_location as mcl
import utils.training_utils.pairwise_processing as pairwise_processing

date_now = datetime.datetime.now().strftime('%Y-%m-%d')
INTERMEDIATE_DATA_LOCATION = 'intermediate_data/'
TRAIN_DEV_TEST_DATA_LOCATION = 'data/'
NMT_MODEL_DIR = 'model/'
SENTENCEPIECE_MODEL_DIR = 'model/sentencepiece_models/'
TRAIN_LOG_FILE = 'intermediate_data/anuvaad_training_log_file.txt'

logger = init_logger(TRAIN_LOG_FILE)

def onmt_train(f_in):
    try:
        logger.info("onmt_train: Preprrocessing starting")
        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200 \
                  -save_data {4}'.format(f_in['train_src'],f_in['train_tgt'],f_in['valid_src'],f_in['valid_tgt'],f_in['nmt_processed_data']))
        logger.info("preprocessing done, starting training for {}- epochs".format(f_in['epoch']))

        os.system('rm -f {0} {1} {2} {3}'.format(f_in['train_src'],f_in['train_tgt'],f_in['valid_src'],f_in['valid_tgt']))
        logger.info("removed files, starting training for epoch:{}".format(f_in['epoch']))
        
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
                  -encoder_type transformer -decoder_type transformer -position_encoding -train_steps {2} -max_generator_batches 2 -dropout 0.1  \
                  -batch_size 6000 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam \
                  -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 \
                  -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0 &'.format(f_in['nmt_processed_data'],f_in['nmt_model_path'],f_in['epoch']))

    except Exception as e:
        logger.error("error in onmt_train utils-anuvaad script: {}".format(e))

def incremental_training(src_sp_model,tgt_sp_model,train_from_model):
    "need to change it as per latest flow:in process"
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_hindi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_hindi','model_en-hi_inc_exp-5.10_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_hindi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_hindi'))
            print("folder created at {}".format(model_intermediate_folder))
        hindi_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_train_tok.txt')
        hindi_dev_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_dev_tok.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok.txt')
        hindi_encoded_file = os.path.join(model_master_train_folder, 'hindi_train_final.txt')
        hindi_dev_encoded_file = os.path.join(model_master_train_folder, 'hindi_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_inc_exp-5.10_{}'.format(date_now))

        print("Exp-5.10 incremental training")
        print("src sp:{},tgt sp:{}, from:{}".format(src_sp_model,tgt_sp_model,train_from_model))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['HINDI_TRAIN_FILE'], hindi_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['DEV_HINDI'], hindi_dev_tokenized_file))
        logger.info("english-hindi, hindi train and dev corpus tokenized")
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['DEV_ENGLISH'], english_dev_tokenized_file))
        logger.info("english-hindi, english train, dev,test corpus tokenized")
        print("tokenization done")

        sp.encode_as_pieces(tgt_sp_model,hindi_tokenized_file,hindi_encoded_file)
        sp.encode_as_pieces(tgt_sp_model,hindi_dev_tokenized_file,hindi_dev_encoded_file)
        logger.info("hindi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(src_sp_model,english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(src_sp_model,english_dev_tokenized_file,english_dev_encoded_file)
        logger.info("english-train,dev,test file encoded and final stored in data folder")
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 200 -tgt_seq_length 200 -save_data {4}'.format(english_encoded_file,hindi_encoded_file,english_dev_encoded_file,hindi_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        os.system('python ./embeddings_to_torch.py -emb_file_enc ~/glove/glove.6B.300d.txt -emb_file_dec ~/glove/cc.hi.300.vec -dict_file {0} -output_file {1}'.format(nmt_processed_data+'.vocab.pt',os.path.join(model_master_train_folder,'embeddings_eng_hin')))
        print("glove embedding done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding -train_from {2} -train_steps 250000  -max_generator_batches 2 -dropout 0.1 -batch_size 6000 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 5000 -save_checkpoint_steps 50000 -world_size 1 -gpu_ranks 0'.format(nmt_processed_data,nmt_model_path,train_from_model))

    except Exception as e:
        print(e)
        logger.info("error in english_hindi anuvaad script: {}".format(e))

def english_hindi_sharevocab_experiments():

    try:
        sp_model_prefix_en_hi_sv = 'en_hi_exp-2-sv-{}-48k'.format(date_now)
        # sp_model_prefix_english = 'en_exp-5.4-{}-24k'.format(date_now)
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_train_folder = os.path.join(TRAIN_DEV_TEST_DATA_LOCATION, 'english_hindi')
        nmt_model_path = os.path.join(NMT_MODEL_DIR, 'english_hindi','model_en_hi_exp-2-sv_{}-model'.format(date_now))
        if not any([os.path.exists(model_intermediate_folder),os.path.exists(model_master_train_folder),os.path.exists(os.path.join(NMT_MODEL_DIR, 'english_hindi'))]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_train_folder)
            os.makedirs(os.path.join(NMT_MODEL_DIR, 'english_hindi'))
            print("folder created at {}".format(model_intermediate_folder))
        hindi_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_train_tok.txt')
        hindi_dev_tokenized_file = os.path.join(model_intermediate_folder, 'hindi_dev_tok.txt')
        english_tokenized_file = os.path.join(model_intermediate_folder, 'english_train_tok.txt')
        english_dev_tokenized_file = os.path.join(model_intermediate_folder, 'english_dev_tok.txt')
        merged_src_tgt = os.path.join(model_intermediate_folder, 'merged_src_tgt.txt')
        # english_test_Gen_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_Gen_tok.txt')
        # english_test_LC_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_LC_tok.txt')
        # english_test_TB_tokenized_file = os.path.join(model_intermediate_folder, 'english_test_TB_tok.txt')
        hindi_encoded_file = os.path.join(model_master_train_folder, 'hindi_train_final.txt')
        hindi_dev_encoded_file = os.path.join(model_master_train_folder, 'hindi_dev_final.txt')
        english_encoded_file = os.path.join(model_master_train_folder, 'english_train_final.txt')
        english_dev_encoded_file = os.path.join(model_master_train_folder, 'english_dev_final.txt')
        # english_test_Gen_encoded_file = os.path.join(model_master_train_folder, 'english_test_Gen_final.txt')
        # english_test_LC_encoded_file = os.path.join(model_master_train_folder, 'english_test_LC_final.txt')
        # english_test_TB_encoded_file = os.path.join(model_master_train_folder, 'english_test_TB_final.txt')
        nmt_processed_data = os.path.join(model_master_train_folder, 'processed_data_en_hi_exp-2-sv_{}'.format(date_now))

        print("en_hi_exp-2-sv training")
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['HINDI_TRAIN_FILE'], hindi_tokenized_file))
        os.system('python ./tools/indic_tokenize.py {0} {1} hi'.format(mcl.english_hindi['DEV_HINDI'], hindi_dev_tokenized_file))
        logger.info("english-hindi, hindi train and dev corpus tokenized")
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['ENGLISH_TRAIN_FILE'], english_tokenized_file))
        os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['DEV_ENGLISH'], english_dev_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_GEN'], english_test_Gen_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_LC'], english_test_LC_tokenized_file))
        # os.system('perl ./tools/tokenizer.perl <{0}> {1}'.format(mcl.english_hindi['TEST_ENGLISH_TB'], english_test_TB_tokenized_file))
        logger.info("english-hindi, english train, dev,test corpus tokenized")
        os.system('cat {0} {1} > {2}'.format(english_tokenized_file, hindi_tokenized_file, merged_src_tgt))



        sp.train_spm(merged_src_tgt,sp_model_prefix_en_hi_sv, 48000, 'bpe')
        print("sentencepiece model shared trained")
        # sp.train_spm(english_tokenized_file,sp_model_prefix_english, 24000, 'bpe')
        # logger.info("sentencepiece model english trained")

        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_en_hi_sv+'.model')),hindi_tokenized_file,hindi_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_en_hi_sv+'.model')),hindi_dev_tokenized_file,hindi_dev_encoded_file)
        logger.info("hindi-train file and dev encoded and final stored in data folder")
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_en_hi_sv+'.model')),english_tokenized_file,english_encoded_file)
        sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_en_hi_sv+'.model')),english_dev_tokenized_file,english_dev_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_Gen_tokenized_file,english_test_Gen_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_LC_tokenized_file,english_test_LC_encoded_file)
        # sp.encode_as_pieces(os.path.join(SENTENCEPIECE_MODEL_DIR, (sp_model_prefix_english+'.model')),english_test_TB_tokenized_file,english_test_TB_encoded_file)
        
        print("english-train,dev,test file encoded and final stored in data folder")

        os.system('python preprocess.py -train_src {0} -train_tgt {1} -valid_src {2} -valid_tgt {3} -src_seq_length 150 -tgt_seq_length 150 -share_vocab -save_data {4}'.format(english_encoded_file,hindi_encoded_file,english_dev_encoded_file,hindi_dev_encoded_file,nmt_processed_data))
        print("preprocessing done")
        # os.system('python ./embeddings_to_torch.py -emb_file_enc ~/glove/glove.6B.300d.txt -emb_file_dec ~/glove/cc.hi.300.vec -dict_file {0} -output_file {1}'.format(nmt_processed_data+'.vocab.pt',os.path.join(model_master_train_folder,'embeddings_eng_hin')))
        # print("glove embedding done")
        os.system('nohup python train.py -data {0} -save_model {1} -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 -share_embeddings -encoder_type transformer -decoder_type transformer -position_encoding -train_steps 150000  -max_generator_batches 2 -dropout 0.1 -batch_size 6000 -batch_type tokens -normalization tokens  -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 0.25 -max_grad_norm 0 -param_init 0  -param_init_glorot  -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 -world_size 1 -gpu_ranks 0'.format(nmt_processed_data,nmt_model_path))

    except Exception as e:
        print(e)
        logger.info("error in english_hindi anuvaad script: {}".format(e))