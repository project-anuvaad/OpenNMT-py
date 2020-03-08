"Note:Original Corpus is obtained from data lake"
import sys
import corpus.file_merger as fm
import corpus.file_cleaner as fc
import corpus.helper_functions.format_handler as format_handler
import os
import datetime
import corpus.original_corpus_location as ocl
from onmt.utils.logging import init_logger
import uuid

INTERMEDIATE_DATA_LOCATION = 'corpus/intermediate_data/'
MASTER_DATA_LOCATION = 'corpus/master_corpus'
TRAIN_LOG_FILE = 'intermediate_data/anuvaad_training_log_file.txt'

logger = init_logger(TRAIN_LOG_FILE)

def english_tamil(eng_file,tamil_file):
    try:
        logger.info("English and tamil corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_tamil')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_tamil')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        tamil_merged_file_name = os.path.join(model_intermediate_folder, 'tamil_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        tamil_separated = os.path.join(model_intermediate_folder, 'tamil_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        tamil_dev_separated = os.path.join(model_intermediate_folder, 'tamil_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        tamil_tagged = os.path.join(model_master_folder, 'tamil_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_tamil_tagged = os.path.join(model_master_folder, 'tamil_dev_final'+unique_id+'.txt')
        # test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_tamil_tagged = os.path.join(model_master_folder, 'tamil_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_tamil['FILE_NAMES_ENGLISH']
        # file_names_tamil = ocl.english_tamil['FILE_NAMES_tamil']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_tamil, tamil_merged_file_name)
        # print("original src and tgt file merged successfully")
                

        fc.tab_separated_parllel_corpus(tamil_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, tamil_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, tamil_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(tamil_separated, tamil_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(tamil_dev_separated, dev_tamil_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        # format_handler.tag_number_date_url(ocl.english_tamil['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_tamil['TEST_TAMIL'], test_tamil_tagged)
        # logger.info("test data taggeg and in master folder : Preprocesssing finished !")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,tamil_separated,eng_dev_separated,tamil_dev_separated))

        logger.info("intermediate files are removed successfully: in corpus/scripts/en-tamil")

        return {'ENGLISH_TRAIN_FILE':english_tagged,'TAMIL_TRAIN_FILE':tamil_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_TAMIL':dev_tamil_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_tamil_experiments corpus/scripts- {}".format(e))

def english_hindi():
    "last-18/09/19 model, not using thi function"
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_hindi')
        # model_intermediate_folder = datetime.datetime.now().strftime('%Y-%m-%d')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        # english_merged_lowercased_file_name = os.path.join(model_intermediate_folder, 'english_merged_lowercased_original.txt')
        hindi_merged_file_name = os.path.join(model_intermediate_folder, 'hindi_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        hindi_separated = os.path.join(model_intermediate_folder, 'hindi_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        hindi_tagged = os.path.join(model_master_folder, 'hindi_train_corpus_final.txt')

        # dev_english_lowercased = os.path.join(model_intermediate_folder, 'english_dev_lowercased.txt')
        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_hindi_tagged = os.path.join(model_master_folder, 'hindi_dev_final.txt')
        # test_Gen_english_lowercased = os.path.join(model_intermediate_folder, 'english_test_Gen_lowercased.txt')
        # test_LC_english_lowercased = os.path.join(model_intermediate_folder, 'english_test_LC_lowercased.txt')
        # test_TB_english_lowercased = os.path.join(model_intermediate_folder, 'english_test_TB_lowercased.txt')
        test_Gen_english_tagged = os.path.join(model_master_folder, 'english_test_Gen_final.txt')
        test_LC_english_tagged = os.path.join(model_master_folder, 'english_test_LC_final.txt')
        test_TB_english_tagged = os.path.join(model_master_folder, 'english_test_TB_final.txt')
        test_Gen_hindi_tagged = os.path.join(model_master_folder, 'hindi_test_Gen_final.txt')
        test_LC_hindi_tagged = os.path.join(model_master_folder, 'hindi_test_LC_final.txt')
        test_TB_hindi_tagged = os.path.join(model_master_folder, 'hindi_test_TB_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        file_names_english = ocl.english_hindi['FILE_NAMES_ENGLISH']
        file_names_hindi = ocl.english_hindi['FILE_NAMES_HINDI']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_hindi, hindi_merged_file_name)
        print("original src and tgt file merged successfully")
                

        fc.tab_separated_parllel_corpus(hindi_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        # fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        # print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, hindi_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(hindi_separated, hindi_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_hindi['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['DEV_HINDI'], dev_hindi_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_ENGLISH_GEN'], test_Gen_english_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_ENGLISH_LC'], test_LC_english_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_ENGLISH_TB'], test_TB_english_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_HINDI_GEN'], test_Gen_hindi_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_HINDI_LC'], test_LC_hindi_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_HINDI_TB'], test_TB_hindi_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)

def english_hindi_experiments(eng_file,hindi_file):
    "Exp-5.4: -data same as 5.1 exp...old data+ india kanoon 830k(including 1.5 lakhs names n no learned counsel)+72192k shabkosh, BPE 24k, nolowercasing,pretok,shuffling"
    try:
        logger.info("English and hindi corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_hindi')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        hindi_merged_file_name = os.path.join(model_intermediate_folder, 'hindi_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        hindi_separated = os.path.join(model_intermediate_folder, 'hindi_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        hindi_dev_separated = os.path.join(model_intermediate_folder, 'hindi_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        hindi_tagged = os.path.join(model_master_folder, 'hindi_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_hindi_tagged = os.path.join(model_master_folder, 'hindi_dev_final'+unique_id+'.txt')
        test_Gen_english_tagged = os.path.join(model_master_folder, 'english_test_Gen_final.txt')
        test_LC_english_tagged = os.path.join(model_master_folder, 'english_test_LC_final.txt')
        test_Gen_hindi_tagged = os.path.join(model_master_folder, 'hindi_test_Gen_final.txt')
        test_LC_hindi_tagged = os.path.join(model_master_folder, 'hindi_test_LC_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_hindi['FILE_NAMES_ENGLISH']
        # file_names_hindi = ocl.english_hindi['FILE_NAMES_HINDI']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_hindi, hindi_merged_file_name)
        # print("original src and tgt file merged successfully")
                

        fc.tab_separated_parllel_corpus(hindi_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, hindi_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, hindi_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(hindi_separated, hindi_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(hindi_dev_separated, dev_hindi_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_ENGLISH_GEN'], test_Gen_english_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_ENGLISH_LC'], test_LC_english_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_HINDI_GEN'], test_Gen_hindi_tagged)
        format_handler.tag_number_date_url(ocl.english_hindi['TEST_HINDI_LC'], test_LC_hindi_tagged)
        logger.info("test data taggeg and in master folder : Preprocesssing finished !")

        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,hindi_separated,eng_dev_separated,hindi_dev_separated))
        logger.info("intermediate files are removed successfully: in corpus/scripts/en-hi") 

        return {'ENGLISH_TRAIN_FILE':english_tagged,'HINDI_TRAIN_FILE':hindi_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_HINDI':dev_hindi_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_hindi_experiments corpus/scripts- {}".format(e))

def english_gujarati(eng_file,gujarati_file):
    try:
        logger.info("English and gujarati corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_gujarati')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_gujarati')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        gujarati_merged_file_name = os.path.join(model_intermediate_folder, 'gujarati_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        gujarati_separated = os.path.join(model_intermediate_folder, 'gujarati_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        gujarati_dev_separated = os.path.join(model_intermediate_folder, 'gujarati_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        gujarati_tagged = os.path.join(model_master_folder, 'gujarati_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_gujarati_tagged = os.path.join(model_master_folder, 'gujarati_dev_final'+unique_id+'.txt')
        # test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_gujarati_tagged = os.path.join(model_master_folder, 'gujarati_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_gujarati['FILE_NAMES_ENGLISH']
        # file_names_gujarati = ocl.english_gujarati['FILE_NAMES_GUJARATI']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_gujarati, gujarati_merged_file_name)
        # print("original src and tgt file merged successfully")
                
        fc.tab_separated_parllel_corpus(gujarati_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, gujarati_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, gujarati_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(gujarati_separated, gujarati_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(gujarati_dev_separated, dev_gujarati_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        # format_handler.tag_number_date_url(ocl.english_gujarati['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_gujarati['TEST_GUJARATI'], test_gujarati_tagged)
        # logger.info("test data taggeg and in master folder : Preprocesssing finished !")
        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,gujarati_separated,eng_dev_separated,gujarati_dev_separated))

        logger.info("intermediate files are removed successfully: in corpus/scripts")          

        return {'ENGLISH_TRAIN_FILE':english_tagged,'GUJARATI_TRAIN_FILE':gujarati_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_GUJARATI':dev_gujarati_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_gujarati corpus/scripts- {}".format(e))
    

def english_bengali(eng_file,bengali_file):
    try:
        logger.info("English and bengali corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_bengali')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_bengali')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        bengali_merged_file_name = os.path.join(model_intermediate_folder, 'bengali_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        bengali_separated = os.path.join(model_intermediate_folder, 'bengali_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        bengali_dev_separated = os.path.join(model_intermediate_folder, 'bengali_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        bengali_tagged = os.path.join(model_master_folder, 'bengali_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_bengali_tagged = os.path.join(model_master_folder, 'bengali_dev_final'+unique_id+'.txt')
        # test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_bengali_tagged = os.path.join(model_master_folder, 'bengali_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_bengali['FILE_NAMES_ENGLISH']
        # file_names_bengali = ocl.english_bengali['FILE_NAMES_BENGALI']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_bengali, bengali_merged_file_name)
        # print("original src and tgt file merged successfully")
                
        fc.tab_separated_parllel_corpus(bengali_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, bengali_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, bengali_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(bengali_separated, bengali_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(bengali_dev_separated, dev_bengali_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        # format_handler.tag_number_date_url(ocl.english_bengali['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_bengali['TEST_BENGALI'], test_bengali_tagged)
        # logger.info("test data taggeg and in master folder : Preprocesssing finished !")
        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,bengali_separated,eng_dev_separated,bengali_dev_separated))

        logger.info("intermediate files are removed successfully: in corpus/scripts")          

        return {'ENGLISH_TRAIN_FILE':english_tagged,'BENGALI_TRAIN_FILE':bengali_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_BENGALI':dev_bengali_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_bengali corpus/scripts/en-bengali- {}".format(e))
    
def english_marathi(eng_file,marathi_file):
    try:
        logger.info("English and marathi corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_marathi')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_marathi')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        marathi_merged_file_name = os.path.join(model_intermediate_folder, 'marathi_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        marathi_separated = os.path.join(model_intermediate_folder, 'marathi_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        marathi_dev_separated = os.path.join(model_intermediate_folder, 'marathi_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        marathi_tagged = os.path.join(model_master_folder, 'marathi_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_marathi_tagged = os.path.join(model_master_folder, 'marathi_dev_final'+unique_id+'.txt')
        # test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_marathi_tagged = os.path.join(model_master_folder, 'marathi_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_marathi['FILE_NAMES_ENGLISH']
        # file_names_marathi = ocl.english_marathi['FILE_NAMES_marathi']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_marathi, marathi_merged_file_name)
        # print("original src and tgt file merged successfully")
                
        fc.tab_separated_parllel_corpus(marathi_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, marathi_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, marathi_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(marathi_separated, marathi_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(marathi_dev_separated, dev_marathi_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        # format_handler.tag_number_date_url(ocl.english_marathi['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_marathi['TEST_MARATHI'], test_marathi_tagged)
        # logger.info("test data taggeg and in master folder : Preprocesssing finished !")
        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,marathi_separated,eng_dev_separated,marathi_dev_separated))

        logger.info("intermediate files are removed successfully: in corpus/scripts")          

        return {'ENGLISH_TRAIN_FILE':english_tagged,'MARATHI_TRAIN_FILE':marathi_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_MARATHI':dev_marathi_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_marathi_experiments corpus/scripts/en-mr- {}".format(e))
    

def english_kannada(eng_file,kannada_file):
    try:
        logger.info("English and kannada corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_kannada')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_kannada')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        kannada_merged_file_name = os.path.join(model_intermediate_folder, 'kannada_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        kannada_separated = os.path.join(model_intermediate_folder, 'kannada_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        kannada_dev_separated = os.path.join(model_intermediate_folder, 'kannada_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        kannada_tagged = os.path.join(model_master_folder, 'kannada_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_kannada_tagged = os.path.join(model_master_folder, 'kannada_dev_final'+unique_id+'.txt')
        # test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_kannada_tagged = os.path.join(model_master_folder, 'kannada_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_kannada['FILE_NAMES_ENGLISH']
        # file_names_kannada = ocl.english_kannada['FILE_NAMES_KANNADA']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_kannada, kannada_merged_file_name)
        # print("original src and tgt file merged successfully")
                
        fc.tab_separated_parllel_corpus(kannada_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, kannada_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, kannada_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(kannada_separated, kannada_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(kannada_dev_separated, dev_kannada_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        # format_handler.tag_number_date_url(ocl.english_kannada['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_kannada['TEST_KANNADA'], test_kannada_tagged)
        # logger.info("test data taggeg and in master folder : Preprocesssing finished !")
        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,kannada_separated,eng_dev_separated,kannada_dev_separated))

        logger.info("intermediate files are removed successfully: in corpus/scripts")          

        return {'ENGLISH_TRAIN_FILE':english_tagged,'KANNADA_TRAIN_FILE':kannada_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_KANNADA':dev_kannada_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_kannada corpus/scripts/en-kannada- {}".format(e))
    

def english_telugu(eng_file,telugu_file):
    try:
        logger.info("English and telugu corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_telugu')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_telugu')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        telugu_merged_file_name = os.path.join(model_intermediate_folder, 'telugu_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        telugu_separated = os.path.join(model_intermediate_folder, 'telugu_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        telugu_dev_separated = os.path.join(model_intermediate_folder, 'telugu_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        telugu_tagged = os.path.join(model_master_folder, 'telugu_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_telugu_tagged = os.path.join(model_master_folder, 'telugu_dev_final'+unique_id+'.txt')
        # test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_telugu_tagged = os.path.join(model_master_folder, 'telugu_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_telugu['FILE_NAMES_ENGLISH']
        # file_names_telugu = ocl.english_telugu['FILE_NAMES_TELUGU']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_telugu, telugu_merged_file_name)
        # print("original src and tgt file merged successfully")
                
        fc.tab_separated_parllel_corpus(telugu_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, telugu_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, telugu_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(telugu_separated, telugu_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(telugu_dev_separated, dev_telugu_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        # format_handler.tag_number_date_url(ocl.english_telugu['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_telugu['TEST_TELUGU'], test_telugu_tagged)
        # logger.info("test data taggeg and in master folder : Preprocesssing finished !")
        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,telugu_separated,eng_dev_separated,telugu_dev_separated))

        logger.info("intermediate files are removed successfully: in corpus/scripts")          

        return {'ENGLISH_TRAIN_FILE':english_tagged,'TELUGU_TRAIN_FILE':telugu_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_TELUGU':dev_telugu_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_telugu corpus/scripts/en-telugu- {}".format(e))
    

def english_malayalam(eng_file,malayalam_file):
    try:
        logger.info("English and malayalam corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_malayalam')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_malayalam')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        malayalam_merged_file_name = os.path.join(model_intermediate_folder, 'malayalam_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        malayalam_separated = os.path.join(model_intermediate_folder, 'malayalam_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        malayalam_dev_separated = os.path.join(model_intermediate_folder, 'malayalam_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        malayalam_tagged = os.path.join(model_master_folder, 'malayalam_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_malayalam_tagged = os.path.join(model_master_folder, 'malayalam_dev_final'+unique_id+'.txt')
        # test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_malayalam_tagged = os.path.join(model_master_folder, 'malayalam_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_malayalam['FILE_NAMES_ENGLISH']
        # file_names_malayalam = ocl.english_malayalam['FILE_NAMES_MALAYALAM']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_malayalam, malayalam_merged_file_name)
        # print("original src and tgt file merged successfully")
                
        fc.tab_separated_parllel_corpus(malayalam_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, malayalam_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, malayalam_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(malayalam_separated, malayalam_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(malayalam_dev_separated, dev_malayalam_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        # format_handler.tag_number_date_url(ocl.english_malayalam['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_malayalam['TEST_MALAYALAM'], test_malayalam_tagged)
        # logger.info("test data taggeg and in master folder : Preprocesssing finished !")
        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,malayalam_separated,eng_dev_separated,malayalam_dev_separated))

        logger.info("intermediate files are removed successfully: in corpus/scripts/en-malayalam")          

        return {'ENGLISH_TRAIN_FILE':english_tagged,'MALAYALAM_TRAIN_FILE':malayalam_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_MALAYALAM':dev_malayalam_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_malayalam corpus/scripts/en-malayalam- {}".format(e))
    

def english_punjabi(eng_file,punjabi_file):
    try:
        logger.info("English and punjabi corpus preprocessing: starting")

        unique_id = str(uuid.uuid1())
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_punjabi')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_punjabi')

        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original'+unique_id+'.txt')
        punjabi_merged_file_name = os.path.join(model_intermediate_folder, 'punjabi_merged_original'+unique_id+'.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus'+unique_id+'.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate'+unique_id+'.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file'+unique_id+'.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num'+unique_id+'.txt')
        train_file = os.path.join(model_intermediate_folder, 'train_file'+unique_id+'.txt')
        dev_file = os.path.join(model_intermediate_folder, 'dev_file'+unique_id+'.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated'+unique_id+'.txt')
        punjabi_separated = os.path.join(model_intermediate_folder, 'punjabi_train_separated'+unique_id+'.txt')
        eng_dev_separated = os.path.join(model_intermediate_folder, 'eng_dev_separated'+unique_id+'.txt')
        punjabi_dev_separated = os.path.join(model_intermediate_folder, 'punjabi_dev_separated'+unique_id+'.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final'+unique_id+'.txt')
        punjabi_tagged = os.path.join(model_master_folder, 'punjabi_train_corpus_final'+unique_id+'.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final'+unique_id+'.txt')
        dev_punjabi_tagged = os.path.join(model_master_folder, 'punjabi_dev_final'+unique_id+'.txt')
        # test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_punjabi_tagged = os.path.join(model_master_folder, 'punjabi_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            logger.info("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        # file_names_english = ocl.english_punjabi['FILE_NAMES_ENGLISH']
        # file_names_punjabi = ocl.english_punjabi['FILE_NAMES_MALAYALAM']
        # fm.file_merger(file_names_english, english_merged_file_name)
        # fm.file_merger(file_names_punjabi, punjabi_merged_file_name)
        # print("original src and tgt file merged successfully")
                
        fc.tab_separated_parllel_corpus(punjabi_file, eng_file, tab_sep_out_file)
        logger.info("tab separated corpus created")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        logger.info("duplicates removed from combined corpus")
        logger.info(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        logger.info("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        logger.info("hindi number replaced")

        fc.split_into_train_validation(replaced_hindi_number_file_name,train_file,dev_file,0.995)
        logger.info("splitted file into train and validation set")

        fc.separate_corpus(0, train_file, eng_separated)
        fc.separate_corpus(1, train_file, punjabi_separated)
        fc.separate_corpus(0, dev_file, eng_dev_separated)
        fc.separate_corpus(1, dev_file, punjabi_dev_separated)
        logger.info("corpus separated into src and tgt for training and validation")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(punjabi_separated, punjabi_tagged)
        format_handler.tag_number_date_url(eng_dev_separated, dev_english_tagged)
        format_handler.tag_number_date_url(punjabi_dev_separated, dev_punjabi_tagged)

        logger.info("url,num and date tagging done, corpus in master folder")
        # format_handler.tag_number_date_url(ocl.english_punjabi['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_punjabi['TEST_PUNJABI'], test_punjabi_tagged)
        # logger.info("test data taggeg and in master folder : Preprocesssing finished !")
        os.system('rm -f {0} {1} {2} {3} {4} {5} {6} {7} {8} {9}'.format(tab_sep_out_file,tab_sep_out_file_no_duplicate,shuffled_tab_sep_file,\
                  replaced_hindi_number_file_name,train_file,dev_file,eng_separated,punjabi_separated,eng_dev_separated,punjabi_dev_separated))

        logger.info("intermediate files are removed successfully: in corpus/scripts/en-punjabi")          

        return {'ENGLISH_TRAIN_FILE':english_tagged,'PUNJABI_TRAIN_FILE':punjabi_tagged, 'DEV_ENGLISH':dev_english_tagged,'DEV_PUNJABI':dev_punjabi_tagged, \
               'unique_id':unique_id}

    except Exception as e:
        logger.error("error in english_punjabi corpus/scripts/en-punjabi- {}".format(e))
    

def file_shuffler():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'file_shuffler')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_corpus.txt')
        hindi_separated = os.path.join(model_intermediate_folder, 'hindi_corpus.txt')

        if not any ([os.path.exists(model_intermediate_folder)]):
            os.makedirs(model_intermediate_folder)
            print("folder created at {}".format(model_intermediate_folder))                

        fc.tab_separated_parllel_corpus("7062e708-2a87-4a08-a167-a6f3aa7f5cc4_Hindi_target.txt", "7062e708-2a87-4a08-a167-a6f3aa7f5cc4_english_source.txt", tab_sep_out_file)
        print("tab separated corpus created")
        
        format_handler.shuffle_file(tab_sep_out_file,shuffled_tab_sep_file)
        print("tab_sep_file_shuffled_successfully!")

        fc.separate_corpus(0, shuffled_tab_sep_file, eng_separated)
        fc.separate_corpus(1, shuffled_tab_sep_file, hindi_separated)
        print("corpus separated into src and tgt")

    except Exception as e:
        print(e)

# if __name__ == '__main__':
#     if sys.argv[1] == "english-tamil":
#         english_tamil()
#     elif sys.argv[1] == "english-hindi":
#         english_hindi()
#     elif sys.argv[1] == "english-gujrati":
#         english_gujrati() 
#     elif sys.argv[1] == "english-bengali":
#         english_bengali()
#     elif sys.argv[1] == "english-marathi":
#         english_marathi() 
#     elif sys.argv[1] == "english-kannada":
#         english_kannada()  
#     elif sys.argv[1] == "english-telgu":
#         english_telgu()
#     elif sys.argv[1] == "english-malayalam":
#         english_malayalam()   
#     elif sys.argv[1] == "english-punjabi":
#         english_punjabi() 
#     elif sys.argv[1] == "english-hindi-exp":
#         english_hindi_experiments()   
#     elif sys.argv[1] == "english-hindi-exp_5.10":
#         english_hindi_exp_5_10()   
#     elif sys.argv[1] == "ik_5_4_shuffle_for_graders":
#         ik_5_4_shuffle_for_graders()                           
#     else:
#         print("invalid request", sys.argv)
