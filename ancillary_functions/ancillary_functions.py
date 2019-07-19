def handle_single_token(token):
    if isfloat(token):
        return replace_from_LC_table(token)
    elif token.isalnum():
        "in progress"
    else:
        print("returning token as it is")
        return token    
    "if token in alpha-numeric+ sp character"        

def replace_from_LC_table(token):
    hindi_number=list()
    for char in token:
        if char.isdigit():
            with open("lookup_table.txt", "r") as f:
                            for line in f:
                                if line.startswith(char):
                                    char = line.split('|||')[1].strip() 

        hindi_number.append(char) 
    s = [str(i) for i in hindi_number] 
    res = ("".join(s)) 
    return res 

def isfloat(str):
    try: 
        float(str)
    except ValueError: 
        return False
    return True

def capture_prefix_suffix(text):
    prefix = text[0]
    suffix = text[-1] 
    if (prefix.isalpha() or prefix.isdigit()) and (suffix.isalpha() or suffix.isdigit() or suffix == '.'):
        prefix = ""
        suffix = ""
        translation_text = text
    elif (prefix.isalpha() or prefix.isdigit()) and (suffix.isalpha()== False and suffix.isdigit()==False and suffix != '.'): 
        prefix = ""
        translation_text = text[0:]
    elif (prefix.isalpha()==False or prefix.isdigit()==False) and (suffix.isalpha()== False and suffix.isdigit()==False and suffix != '.'):
        translation_text = text[1:-1]  
    elif (prefix.isalpha()==False or prefix.isdigit()==False) and (suffix.isalpha() or suffix.isdigit() or suffix == '.'):  
        suffix = ""
        translation_text = text[1:]     
    print(prefix,suffix,translation_text)
    return prefix,suffix,translation_text

