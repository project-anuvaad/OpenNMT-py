import sys
import file_merger as fm
import file_cleaner as fc
import helper_functions.format_handler as format_handler


def english_tamil():
    try:
        english_merged_file_name = 'corpus/intermediate_data/210819_eng_tamil/eng_merged_210819.txt'
        tamil_merged_file_name = 'corpus/intermediate_data/210819_eng_tamil/tamil_merged_210819.txt'
        tab_sep_out_file = 'corpus/intermediate_data/210819_eng_tamil/tab_sep_corpus.txt'
        tab_sep_out_file_no_duplicate = 'corpus/intermediate_data/210819_eng_tamil/tab_sep_corpus_no_duplicate.txt'
        eng_separated ='corpus/intermediate_data/210819_eng_tamil/eng_separated.txt'
        tamil_separated ='corpus/intermediate_data/210819_eng_tamil/tamil_separated.txt'
        tagged_english = 'corpus/master_corpus/210819_eng_tamil/tagged_english.txt'
        tagged_tamil = 'corpus/master_corpus/210819_eng_tamil/tagged_tamil.txt'
        dev_eng_file = 'corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.dev.en'
        dev_tamil_file = 'corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.dev.ta'
        test_eng_file ='corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.test.en'
        test_tamil_file = 'corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.test.ta'

        file_names_english = ['corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.train.en','corpus/original_data/english_tamil/SC-translated-from-google-210819/1566367056_eng_filtered.txt']
        file_names_tamil = ['corpus/original_data/english_tamil/en-ta-parallel-v2_UFAL/corpus.bcn.train.ta','corpus/original_data/english_tamil/SC-translated-from-google-210819/1566367056_tam_filtered.txt']
        
        fm.file_merger(file_names_english,english_merged_file_name)
        fm.file_merger(file_names_tamil,tamil_merged_file_name)
        print("original src and tgt file merged successfully")

        fc.tab_separated_parllel_corpus(tamil_merged_file_name,english_merged_file_name,tab_sep_out_file)
        print("tab separated corpus created")
        fc.drop_duplicate(tab_sep_out_file,tab_sep_out_file_no_duplicate)
        print("duplicates removed from combined corpus")

        fc.separate_corpus(0,tab_sep_out_file_no_duplicate,eng_separated)
        fc.separate_corpus(1,tab_sep_out_file_no_duplicate,tamil_separated)
        print("corpus separated into src and tgt")

        format_handler.tag_number_date_url(eng_separated,tagged_english)
        format_handler.tag_number_date_url(tamil_separated,tagged_tamil)
        print("url and date tagging done, corpus in master folder")

        format_handler.tag_number_date_url(dev_eng_file,'corpus/master_corpus/210819_eng_tamil/dev_eng_tagged.txt')
        format_handler.tag_number_date_url(dev_tamil_file,'corpus/master_corpus/210819_eng_tamil/dev_tamil_tagged.txt')
        format_handler.tag_number_date_url(test_eng_file,'corpus/master_corpus/210819_eng_tamil/test_eng_tagged.txt')
        format_handler.tag_number_date_url(test_tamil_file,'corpus/master_corpus/210819_eng_tamil/test_tamil_tagged.txt')

        
    except Exception as e:
        print(e)

def english_hindi():
    "in progress"

if __name__ == '__main__':
    if sys.argv[1] == "english-tamil":
        english_tamil()
    elif sys.argv[1] == "english-hindi":
        english_hindi()
    else:
        print("invalid request",sys.argv)        