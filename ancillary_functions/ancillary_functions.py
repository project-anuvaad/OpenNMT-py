import re
from indic_transliteration import sanscript
from indic_transliteration.sanscript import SchemeMap, SCHEMES, transliterate
from onmt.utils.logging import logger

def handle_single_token(token):
    if isfloat(token):
        return replace_from_LC_table(token)
    elif token.isalnum():
        logger.info("transliterating alphanum")
        return transliterate_text(token)

    elif len(token) > 1 and token_is_alphanumeric_char(token):
        prefix,suffix,translation_text = separate_alphanumeric_and_symbol(token)
        translation_text = transliterate_text(translation_text)
        return prefix+translation_text+suffix
    elif token.isalpha() and len(token)==1:
        return transliterate_text(token)    
                
    else:
        logger.info("returning token as it is")
        return token      

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

def token_is_alphanumeric_char(token):
    "checking if single token consists of alphanumeric and symbolic characters. But, symbol only at the begining and end are considerd"
    if re.match(r'^[\w]+$', token) is None:
        return True

def separate_alphanumeric_and_symbol(text):
    # print(re.sub(r"^\W+|\W+$", "", text))    
    start = re.match(r"^\W+|\W+$", text)
    end = re.match(r'.*?([\W]+)$', text)
    translation_text = re.sub(r"^\W+|\W+$", "", text)        
    if start:
        start = start.group(0)
        if start.endswith('(') and translation_text[0].isalnum() and translation_text[1]== ')':
            start = start + transliterate_text(translation_text[0]) + translation_text[1]+'.'
            translation_text = translation_text[2:]
            start_residual_part = re.match(r"^\W+|\W+$", translation_text)
            # print("1",translation_text)    
            if start_residual_part:
                start_residual_part = start_residual_part.group(0)
                start = start+start_residual_part
                translation_text = re.sub(r"^\W+|\W+$", "", translation_text) 
                # print("2",translation_text)     

    else:
        start = ""           
    if end:
        end = end.group(1)
        if end.startswith('.'):
            end = end[1:]
            translation_text = translation_text + '.' 
    else:
        end = ""            
    
    print(start,end,translation_text)     
    return start,end,translation_text

def transliterate_text(text):
    print(transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI))
    return transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)


# match = re.search(r'\(?([0-9A-Za-z]+)\)?', token)
        # print(match.group(1))
        # x = re.sub(r'\(?([0-9A-Za-z]+)\)?',"chink lives in gwlr" ,token)  
