## merge files (all src and all tgt ) -vertical join (step 2 after IITB split)

## From IITB corpus currently we are keeping the data from these groups -chats, Movie Dialogs, general,Hi-Eng Word-Linkage,Admin Dictionary,
#  Admin Examples,Admin Definitions, ted talks, Indic Multi-Parallel, JudicialI, Govt Websites, Book Translations, 
## aditional data- law commison, letters 

def file_merger(file_names,merged_file_location):
    with open("{0}".format(merged_file_location), 'w') as outfile:
        for fname in file_names:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
             

file_names_english = ['corpus/IITB_splitted_data/en-set-430013-434711', 'corpus/IITB_splitted_data/en-set-434711-438933','corpus/IITB_splitted_data/en-set-438933-712818','corpus/IITB_splitted_data/en-set-712818-887993',
             'corpus/IITB_splitted_data/en-set-887993-954457','corpus/IITB_splitted_data/en-set-954457-1001292','corpus/IITB_splitted_data/en-set-1001292-1047815','corpus/IITB_splitted_data/en-set-1047815-1090398',
             'corpus/IITB_splitted_data/en-set-1090398-1100747','corpus/IITB_splitted_data/en-set-1100747-1105754','corpus/IITB_splitted_data/en-set-1105754-1109481','corpus/IITB_splitted_data/en-set-1109481-1232841',
             'corpus/IITB_splitted_data/en-set-1232841-1265704','corpus/IITB_splitted_data/en-set-1265704-1492827','corpus/IITB_splitted_data/en-set-1492827-1561840','corpus/original_data/tgt-law_commision100519.txt','corpus/original_data/english-letters-28062019.txt',
             'corpus/original_data/SC_specific_eng_nonum_020819.txt']

file_names_hindi = ['corpus/IITB_splitted_data/hi-set-430013-434711', 'corpus/IITB_splitted_data/hi-set-434711-438933','corpus/IITB_splitted_data/hi-set-438933-712818','corpus/IITB_splitted_data/hi-set-712818-887993',
             'corpus/IITB_splitted_data/hi-set-887993-954457','corpus/IITB_splitted_data/hi-set-954457-1001292','corpus/IITB_splitted_data/hi-set-1001292-1047815','corpus/IITB_splitted_data/hi-set-1047815-1090398',
             'corpus/IITB_splitted_data/hi-set-1090398-1100747','corpus/IITB_splitted_data/hi-set-1100747-1105754','corpus/IITB_splitted_data/hi-set-1105754-1109481','corpus/IITB_splitted_data/hi-set-1109481-1232841',
             'corpus/IITB_splitted_data/hi-set-1232841-1265704','corpus/IITB_splitted_data/hi-set-1265704-1492827','corpus/IITB_splitted_data/hi-set-1492827-1561840','corpus/original_data/src-law_commision_100519.txt','corpus/original_data/hindi-letters-28062019.txt',
             'corpus/original_data/SC_specific_hin_nonum_020819.txt']

# file_merger(file_names_hindi,'corpus/intermediate_data/merged_testing2108.txt')
# file_merger(file_names_english,'corpus/intermediate_data/merged_testing2108.txt')