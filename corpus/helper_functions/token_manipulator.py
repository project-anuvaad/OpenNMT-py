import csv

def token_finder(input_file,token):
    try:
        out_file = "file_with_{}".format(token)
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
    

def token_list_finder(input_file,token,outfile,no_match_tokens,one_match_tokens):
    try:
        # out_file = "file_with_{}".format("token_list")
        sent_count = 0
        line_numbers = list()
        # outfile = open("{0}".format(out_file), "w")
        with open(input_file) as input_file:
            for num,line in enumerate(input_file,1):
                if any(v in line for v in [token, token.lower(),token.title()]):
                    outfile.write(line)
                    sent_count = sent_count +1
                if sent_count == 2:
                    return     
            if sent_count == 0:
                no_match_tokens.append(token)    
                    # line_numbers.append(num)
            if sent_count == 1:
                one_match_tokens.append(token) 
                        
            # outfile.close()

        # print(line_numbers)
        return None
    except Exception as e:
        print("exception: ",e)
        
def create_tmx_data(tokens):
    one_match_tokens = list ()
    no_match_tokens = list()
    out_file = "file_with_{}".format("token_list")
    outfile = open("{0}".format(out_file), "w")
    tokens_list = tokens
    for token in tokens_list:
        token_list_finder("corpus/original_data/english_bengali/final_bn_source.txt",token,outfile,no_match_tokens,one_match_tokens)
    
    print("all done")    
    print(len(no_match_tokens))
    print("final no match: ",no_match_tokens)    
    print(len(one_match_tokens))
    print(one_match_tokens)

def csv_to_list():
    token_list = list()
    with open('corpus/tmx-english.csv', 'r') as fd:    
        reader = csv.reader(fd)
        for row in reader:
            token_list.append(str(row[0]))
    # print(token_list)
    print(len(token_list))
    print("ddone") 
    return token_list       
        
tokens = csv_to_list()

create_tmx_data(tokens)

def drop_duplicate(inFile,outFile):
    lines_seen = set() # holds lines already seen
    outfile = open("{0}".format(outFile), "w")
    for line in open("{0}".format(inFile), "r"):
        if line not in lines_seen: # not a duplicate
           outfile.write(line)
           lines_seen.add(line)
    outfile.close()
    
# drop_duplicate("file_with_token_list","file_with_token_list_nodup-2")

# index_array = token_finder("corpus/master_corpus/english_hindi/eng_train_corpus_final.txt","" )
# line_extracter_using_index("corpus/master_corpus/english_hindi/eng_train_corpus_final.txt",index_array,"test_en.txt")
# line_extracter_using_index("corpus/master_corpus/english_hindi/hindi_train_corpus_final.txt-1-1",index_array,"test.txt")
# lines_without_token("test3.txt","ADV","test4.txt")
