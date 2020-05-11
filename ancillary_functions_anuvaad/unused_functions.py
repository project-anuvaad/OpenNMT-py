"lookup table"
def lookup_table(model_id,token):
    if model_id in [1]:
        with open("lookup_dictionary_eng_hin.txt",encoding ='utf-16') as xh:
            xlines = xh.readlines()
            for i in range(len(xlines)):
                if xlines[i].split('|||')[0] == token.strip():
                    token = xlines[i].split('|||')[1].strip()
                else:
                    token = ""    
    else:
        token = ""                
    
    return token                