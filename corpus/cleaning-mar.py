def token_finder(input_file,token):
    try:
        out_file = "file_with-19_{}".format("1")
        line_numbers = list()
        outfile = open("{0}".format(out_file), "w")
        with open(input_file) as input_file:
            for num,line in enumerate(input_file,1):
                if token.lower() in line.lower():
                    outfile.write(line)
                    line_numbers.append(num)
            outfile.close()

        print(line_numbers)
        return line_numbers
    except Exception as e:
        print("exception: ",e)

def line_extracter_using_index(input_file,index_arr,out_file):
     try:
         outfile_matched = open("{0}".format(out_file), "w")
         outfile_unmatched = open("{0}-1".format(input_file), "w")
         with open(input_file) as input_file:
             for num, line in enumerate(input_file,1):
                 if num in index_arr:
                     outfile_matched.write(line)
                #  else:
                #      outfile_unmatched.write(line)    
             outfile_matched.close()  
             outfile_unmatched.close()      

     except Exception as e:
         print(e)          

def lines_without_token(input_file,token,out_file):
    try:
        line_numbers = list()
        outfile = open("{0}".format(out_file), "w")
        with open(input_file) as input_file:
            for num,line in enumerate(input_file,1):
                if token.lower() not in line.lower():
                    outfile.write(line)
                    line_numbers.append(num)
            outfile.close()

        print(line_numbers)
        return line_numbers
    except Exception as e:
        print("exception: ",e)

def get_indices_for_same_lines_among_files(file_1,file_2):
    try:
        line_numbers = list()
        open_src = open(file_1,"r")
        read_src = open_src.readlines()

        with open(file_2) as file_2:
            for num, line in enumerate(file_2,1):
                if line in read_src:
                    line_numbers.append(num)
        print(line_numbers)
        return line_numbers
    except Exception as e:
        print(e)
    

# index_array = token_finder("corpus/intermediate_data/file_shuffler/en-ta-19-mar/eng_corpus.txt", "* * *" )
# line_extracter_using_index("corpus/master_corpus/english_hindi/eng_train_corpus_final.txt",index_array,"test_en.txt")
# line_extracter_using_index("corpus/intermediate_data/file_shuffler/en-ta-19-mar/tamil_corpus.txt",index_array,"ta19-1.txt")
# lines_without_token("test3.txt","ADV","test4.txt")

# index_array = token_finder("test_hi.txt-1","एक खंड" )
# line_extracter_using_index("test_en.txt-1",index_array,"positive_persual_en-1.txt")
# line_extracter_using_index("test_hi.txt-1",index_array,"positive_perusual_hi-1.txt")

# indices_1 = get_indices_for_same_lines_among_files('test12.txt',"corpus/master_corpus/english_hindi/hindi_train_corpus_final.txt-1")
# line_extracter_using_index("corpus/master_corpus/english_hindi/eng_train_corpus_final.txt-1",indices_1,"xxxx.txt")
# line_extracter_using_index("corpus/master_corpus/english_hindi/hindi_train_corpus_final.txt-1",indices_1,"yyy.txt")

def clean_file(input_file):
    try:
        out_file = "clean-file-19_{}".format("en")
        outfile = open("{0}".format(out_file), "w")
        with open(input_file) as input_file:
            for line in (input_file):
                line = clean_source(line)
                outfile.write(line)
                outfile.write("\n")
            outfile.close()

        print("done")
    except Exception as e:
        print("exception: ",e)

def clean_source(text):
    text = str(text).strip()
    if text[0:2] == '",':
        text = text.replace('",', '')
    if text[0] == '"':
        text = text.replace('"', '')
    if text[0] == "'":
        text = text.replace("'", '')
    while text.__contains__('\\'):
        text = text.replace('\\', '')
    print('*')
    while text.__contains__('*'):
        text = text.replace('*', '')
    print('*')
    while text.__contains__('","'):
        text = text.replace('","', '')
    print('*')
    while text.__contains__('", "'):
        text = text.replace('", "', '')
    print('*')
    while text.__contains__("',"):
        text = text.replace("',", '')
    print('*')
    while text.__contains__('-- "'):
        text = text.replace('-- "', '')
    print('*')
    while text.__contains__('..."'):
        text = text.replace('..."', '')
    print('*')
    while text.__contains__('""'):
        text = text.replace('""', '"')
    print('*')
    while text.__contains__(']"]'):
        text = text.replace(']"]', '')
    print('*')
    while text.__contains__("']']"):
        text = text.replace("']']", '')
    print('*')
    while text.__contains__('and pay only if you like it.\','):
        text = text.replace('')
    return text


clean_file("corpus/intermediate_data/file_shuffler/en-kn-19-mar/en-19mar.txt")    