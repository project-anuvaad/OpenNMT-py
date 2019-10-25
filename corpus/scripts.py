"Note:Original Corpus is obtained from data lake"
import sys
import file_merger as fm
import file_cleaner as fc
import helper_functions.format_handler as format_handler
import os
import datetime
import original_corpus_location as ocl

INTERMEDIATE_DATA_LOCATION = 'corpus/intermediate_data/'
MASTER_DATA_LOCATION = 'corpus/master_corpus'


def english_tamil():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_tamil')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_tamil')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        tamil_merged_file_name = os.path.join(model_intermediate_folder, 'tamil_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        tamil_separated = os.path.join(model_intermediate_folder, 'tamil_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        tamil_tagged = os.path.join(model_master_folder, 'tamil_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_tamil_tagged = os.path.join(model_master_folder, 'tamil_dev_final.txt')
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        test_tamil_tagged = os.path.join(model_master_folder, 'tamil_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))

        file_names_english = ocl.english_tamil['FILE_NAMES_ENGLISH']
        file_names_tamil = ocl.english_tamil['FILE_NAMES_TAMIL']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_tamil, tamil_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(tamil_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, tamil_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(tamil_separated, tamil_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_tamil['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_tamil['DEV_TAMIL'], dev_tamil_tagged)
        format_handler.tag_number_date_url(ocl.english_tamil['TEST_ENGLISH'], test_english_tagged)
        format_handler.tag_number_date_url(ocl.english_tamil['TEST_TAMIL'], test_tamil_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)


def english_hindi():
    "last-18/09/19 model"
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

def english_hindi_experiments():
    "25/10/2019 experiment 10, Old data + dictionary,BPE, nolowercasing,pretok,shuffling"
    try:
        print("In english_hindi_experiments,scripts,Exp -10")
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_hindi')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        # english_merged_lowercased_file_name = os.path.join(model_intermediate_folder, 'english_merged_lowercased_original.txt')
        hindi_merged_file_name = os.path.join(model_intermediate_folder, 'hindi_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        shuffled_tab_sep_file = os.path.join(model_intermediate_folder, 'shuffled_tab_sep_file.txt')
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
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

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

def english_marathi():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_marathi')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_marathi')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        marathi_merged_file_name = os.path.join(model_intermediate_folder, 'marathi_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        marathi_separated = os.path.join(model_intermediate_folder, 'marathi_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        marathi_tagged = os.path.join(model_master_folder, 'marathi_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_marathi_tagged = os.path.join(model_master_folder, 'marathi_dev_final.txt')
        test_english_tagged = os.path.join(model_master_folder, 'english_test_final.txt')
        # test_marathi_tagged = os.path.join(model_master_folder, 'marathi_test_final.txt')

        if not any ([os.path.exists(model_intermediate_folder),os.path.exists(model_master_folder)]):
            os.makedirs(model_intermediate_folder)
            os.makedirs(model_master_folder)
            print("folder created at {} and {}".format(model_intermediate_folder,model_master_folder))

        file_names_english = ocl.english_marathi['FILE_NAMES_ENGLISH']
        file_names_marathi = ocl.english_marathi['FILE_NAMES_MARATHI']
        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_marathi, marathi_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(marathi_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
        print("hindi number replaced")

        fc.separate_corpus(0, replaced_hindi_number_file_name, eng_separated)
        fc.separate_corpus(1, replaced_hindi_number_file_name, marathi_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, english_tagged)
        format_handler.tag_number_date_url(marathi_separated, marathi_tagged)
        print("url,num and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(ocl.english_marathi['DEV_ENGLISH'], dev_english_tagged)
        format_handler.tag_number_date_url(ocl.english_marathi['DEV_MARATHI'], dev_marathi_tagged)
        format_handler.tag_number_date_url(ocl.english_marathi['TEST_ENGLISH'], test_english_tagged)
        # format_handler.tag_number_date_url(ocl.english_marathi['TEST_MARATHI'], test_marathi_tagged)
        print("test and dev data taggeg and in master folder")

    except Exception as e:
        print(e)

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
    else:
        print("invalid request", sys.argv)
