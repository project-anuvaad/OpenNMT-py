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
TRAIN_LOG_FILE = 'available_models/anuvaad_training_log_file.txt'

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
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        test_tamil_tagged = os.path.join(model_master_folder, 'tamil_test_final.txt')

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
        format_handler.tag_number_date_url(ocl.english_tamil['TEST_ENGLISH'], test_english_tagged)
        format_handler.tag_number_date_url(ocl.english_tamil['TEST_TAMIL'], test_tamil_tagged)
        logger.info("test data taggeg and in master folder : Preprocesssing finished !")

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

def english_gujrati():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_gujrati')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_gujrati')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        gujrati_merged_file_name = os.path.join(model_intermediate_folder, 'gujrati_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        gujrati_separated = os.path.join(model_intermediate_folder, 'gujrati_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        gujrati_tagged = os.path.join(model_master_folder, 'gujrati_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_gujrati_tagged = os.path.join(model_master_folder, 'gujrati_dev_final.txt')
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        test_gujrati_tagged = os.path.join(model_master_folder, 'gujrati_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))

        file_names_english = ocl.english_gujrati['FILE_NAMES_ENGLISH']
        file_names_gujrati = ocl.english_gujrati['FILE_NAMES_GUJRATI']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_gujrati, gujrati_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(gujrati_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, gujrati_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(gujrati_separated, gujrati_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_gujrati['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_gujrati['DEV_GUJRATI'], dev_gujrati_tagged)
        format_handler.tag_number_date_url(ocl.english_gujrati['TEST_ENGLISH'], test_english_tagged)
        format_handler.tag_number_date_url(ocl.english_gujrati['TEST_GUJRATI'], test_gujrati_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)

def english_bengali():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_bengali')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_bengali')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        bengali_merged_file_name = os.path.join(model_intermediate_folder, 'bengali_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        bengali_separated = os.path.join(model_intermediate_folder, 'bengali_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        bengali_tagged = os.path.join(model_master_folder, 'bengali_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_bengali_tagged = os.path.join(model_master_folder, 'bengali_dev_final.txt')
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        test_bengali_tagged = os.path.join(model_master_folder, 'bengali_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))

        file_names_english = ocl.english_bengali['FILE_NAMES_ENGLISH']
        file_names_bengali = ocl.english_bengali['FILE_NAMES_BENGALI']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_bengali, bengali_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(bengali_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, bengali_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(bengali_separated, bengali_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_bengali['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_bengali['DEV_BENGALI'], dev_bengali_tagged)
        format_handler.tag_number_date_url(ocl.english_bengali['TEST_ENGLISH'], test_english_tagged)
        format_handler.tag_number_date_url(ocl.english_bengali['TEST_BENGALI'], test_bengali_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)    

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
        logger.error("error in english_marathi_experiments corpus/scripts- {}".format(e))
    

def english_kannada():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_kannada')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_kannada')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        kannada_merged_file_name = os.path.join(model_intermediate_folder, 'kannada_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        kannada_separated = os.path.join(model_intermediate_folder, 'kannada_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        kannada_tagged = os.path.join(model_master_folder, 'kannada_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_kannada_tagged = os.path.join(model_master_folder, 'kannada_dev_final.txt')
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_kannada_tagged = os.path.join(model_master_folder, 'kannada_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))

        file_names_english = ocl.english_kannada['FILE_NAMES_ENGLISH']
        file_names_kannada = ocl.english_kannada['FILE_NAMES_KANNADA']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_kannada, kannada_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(kannada_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, kannada_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(kannada_separated, kannada_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_kannada['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_kannada['DEV_KANNADA'], dev_kannada_tagged)
        format_handler.tag_number_date_url(ocl.english_kannada['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_kannada['TEST_KANNADA'], test_kannada_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)

def english_telgu():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_telgu')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_telgu')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        telgu_merged_file_name = os.path.join(model_intermediate_folder, 'telgu_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        telgu_separated = os.path.join(model_intermediate_folder, 'telgu_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        telgu_tagged = os.path.join(model_master_folder, 'telgu_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_telgu_tagged = os.path.join(model_master_folder, 'telgu_dev_final.txt')
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_telgu_tagged = os.path.join(model_master_folder, 'telgu_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))

        file_names_english = ocl.english_telgu['FILE_NAMES_ENGLISH']
        file_names_telgu = ocl.english_telgu['FILE_NAMES_TELGU']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_telgu, telgu_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(telgu_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, telgu_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(telgu_separated, telgu_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_telgu['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_telgu['DEV_TELGU'], dev_telgu_tagged)
        format_handler.tag_number_date_url(ocl.english_telgu['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_telgu['TEST_TELGU'], test_telgu_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)

def english_malayalam():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_malayalam')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_malayalam')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        malayalam_merged_file_name = os.path.join(model_intermediate_folder, 'malayalam_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        malayalam_separated = os.path.join(model_intermediate_folder, 'malayalam_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        malayalam_tagged = os.path.join(model_master_folder, 'malayalam_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_malayalam_tagged = os.path.join(model_master_folder, 'malayalam_dev_final.txt')
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_malayalam_tagged = os.path.join(model_master_folder, 'malayalam_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))

        file_names_english = ocl.english_malayalam['FILE_NAMES_ENGLISH']
        file_names_malayalam = ocl.english_malayalam['FILE_NAMES_MALAYALAM']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_malayalam, malayalam_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(malayalam_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, malayalam_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(malayalam_separated, malayalam_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_malayalam['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_malayalam['DEV_MALAYALAM'], dev_malayalam_tagged)
        format_handler.tag_number_date_url(ocl.english_malayalam['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_malayalam['TEST_MALAYALAM'], test_malayalam_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)

def english_punjabi():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_punjabi')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_punjabi')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        punjabi_merged_file_name = os.path.join(model_intermediate_folder, 'punjabi_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        punjabi_separated = os.path.join(model_intermediate_folder, 'punjabi_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        punjabi_tagged = os.path.join(model_master_folder, 'punjabi_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_punjabi_tagged = os.path.join(model_master_folder, 'punjabi_dev_final.txt')
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_punjabi_tagged = os.path.join(model_master_folder, 'punjabi_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))

        file_names_english = ocl.english_punjabi['FILE_NAMES_ENGLISH']
        file_names_punjabi = ocl.english_punjabi['FILE_NAMES_PUNJABI']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_punjabi, punjabi_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(punjabi_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, punjabi_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(punjabi_separated, punjabi_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_punjabi['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_punjabi['DEV_PUNJABI'], dev_punjabi_tagged)
        format_handler.tag_number_date_url(ocl.english_punjabi['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_punjabi['TEST_PUNJABI'], test_punjabi_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)

def english_hindi_exp_5_10():
    "Exp-5.10: -processing 245529k sentences"
    try:
        print("In english_hindi_experiments,scripts,Exp 5.10 eng-hindi")
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi_5.10')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_hindi_5.10')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original_5.10.txt')
        hindi_merged_file_name = os.path.join(model_intermediate_folder, 'hindi_merged_original_5.10.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus_5.10.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate_5.10.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file_5.10.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num_5.10.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated_5.10.txt')
        hindi_separated = os.path.join(model_intermediate_folder, 'hindi_train_separated_5.10.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final_5.10.txt')
        hindi_tagged = os.path.join(model_master_folder, 'hindi_train_corpus_final_5.10.txt')


        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))
        
        file_names_english = ocl.english_hindi['FILE_NAMES_ENGLISH_5.10']
        file_names_hindi = ocl.english_hindi['FILE_NAMES_HINDI_5.10']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_hindi, hindi_merged_file_name)
        print("original src and tgt file merged successfully")
                

        fc.tab_separated_parllel_corpus(hindi_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        print(os.system('wc -l {}'.format(tab_sep_out_file)))
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")
        print(os.system('wc -l {}'.format(tab_sep_out_file_no_duplicate)))
        
        format_handler.shuffle_file(tab_sep_out_file_no_duplicate,shuffled_tab_sep_file)
        print("tab_sep_file_shuffled_successfully!")

        format_handler.replace_hindi_numbers(shuffled_tab_sep_file,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, hindi_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(hindi_separated, hindi_tagged)
        print("url,num and date tagging done, corpus in master folder")

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
