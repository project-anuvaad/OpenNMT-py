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
        english_merged_file_name = 'corpus/intermediate_data/210819_eng_tamil/eng_merged_210819.txt'
        tamil_merged_file_name = 'corpus/intermediate_data/210819_eng_tamil/tamil_merged_210819.txt'
        tab_sep_out_file = 'corpus/intermediate_data/210819_eng_tamil/tab_sep_corpus.txt'
        tab_sep_out_file_no_duplicate = 'corpus/intermediate_data/210819_eng_tamil/tab_sep_corpus_no_duplicate.txt'
        eng_separated = 'corpus/intermediate_data/210819_eng_tamil/eng_separated.txt'
        tamil_separated = 'corpus/intermediate_data/210819_eng_tamil/tamil_separated.txt'
        tagged_english = 'corpus/master_corpus/210819_eng_tamil/tagged_english.txt'
        tagged_tamil = 'corpus/master_corpus/210819_eng_tamil/tagged_tamil.txt'
        dev_eng_file = 'corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.dev.en'
        dev_tamil_file = 'corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.dev.ta'
        test_eng_file = 'corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.test.en'
        test_tamil_file = 'corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.test.ta'

        file_names_english = ['corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.train.en',
                              'corpus/original_data/english_tamil/SC-translated-from-google-210819/1566367056_eng_filtered.txt']
        file_names_tamil = ['corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.train.ta',
                            'corpus/original_data/english_tamil/SC-translated-from-google-210819/1566367056_tam_filtered.txt']

        fm.file_merger(file_names_english, english_merged_file_name)
        fm.file_merger(file_names_tamil, tamil_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(
            tamil_merged_file_name, english_merged_file_name, tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file, tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        fc.separate_corpus(0, tab_sep_out_file_no_duplicate, eng_separated)
        fc.separate_corpus(1, tab_sep_out_file_no_duplicate, tamil_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated, tagged_english)
        format_handler.tag_number_date_url(tamil_separated, tagged_tamil)
        print("url and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(dev_eng_file, 'corpus/master_corpus/210819_eng_tamil/dev_eng_tagged.txt')
        format_handler.tag_number_date_url(dev_tamil_file, 'corpus/master_corpus/210819_eng_tamil/dev_tamil_tagged.txt')
        format_handler.tag_number_date_url(test_eng_file, 'corpus/master_corpus/210819_eng_tamil/test_eng_tagged.txt')
        format_handler.tag_number_date_url(test_tamil_file, 'corpus/master_corpus/210819_eng_tamil/test_tamil_tagged.txt')

    except Exception as e:
        print(e)


def english_hindi():
    try:
        model_intermediate_folder = os.path.join(INTERMEDIATE_DATA_LOCATION, 'english_hindi')
        model_master_folder = os.path.join(MASTER_DATA_LOCATION, 'english_hindi')
        # model_intermediate_folder = datetime.datetime.now().strftime('%Y-%m-%d')
        english_merged_file_name = os.path.join(model_intermediate_folder, 'english_merged_original.txt')
        hindi_merged_file_name = os.path.join(model_intermediate_folder, 'hindi_merged_original.txt')
        tab_sep_out_file = os.path.join(model_intermediate_folder, 'tab_sep_corpus.txt')
        tab_sep_out_file_no_duplicate = os.path.join(model_intermediate_folder, 'tab_sep_corpus_no_duplicate.txt')
        replaced_hindi_number_file_name = os.path.join(model_intermediate_folder, 'corpus_no_hindi_num.txt')
        eng_separated = os.path.join(model_intermediate_folder, 'eng_train_separated.txt')
        hindi_separated = os.path.join(model_intermediate_folder, 'hindi_train_separated.txt')
        english_tagged = os.path.join(model_master_folder, 'eng_train_corpus_final.txt')
        hindi_tagged = os.path.join(model_master_folder, 'hindi_train_corpus_final.txt')

        dev_english_tagged = os.path.join(model_master_folder, 'english_dev_final.txt')
        dev_hindi_tagged = os.path.join(model_master_folder, 'hindi_dev_final.txt')
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

        format_handler.replace_hindi_numbers(tab_sep_out_file_no_duplicate,replaced_hindi_number_file_name)
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

if __name__ == '__main__':
    if sys.argv[1] == "english-tamil":
        english_tamil()
    elif sys.argv[1] == "english-hindi":
        english_hindi()
    elif sys.argv[1] == "english-gujrati":
        english_gujrati() 
    elif sys.argv[1] == "english-bengali":
        english_bengali()       
    else:
        print("invalid request", sys.argv)
