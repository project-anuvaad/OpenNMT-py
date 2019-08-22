import re

def get_numbers(text): 
    try:
        array = re.findall(r'[0-9]+', text)
        s = [str(i) for i in array] 
        res = int("".join(s)) 
        return str(res)
    except:
        return ""
     

def replace_numbers_with_hash(text):
    try:
        text = re.sub(r'\d', '#', text)   
        return text
    except:
        return text
    

def replace_hash_with_original_number(text,num_str):
    try:
        z = 0
        result_str = list()
        for i, v in enumerate(text):
            if z <= i:
                if v == '#' and (len(num_str)> z):
                    v = num_str[z]
                    z +=1
                else:
                    print("xxxxxxxxxxx",v)
            result_str.append(v)
            s = [str(i) for i in result_str] 
            res = str("".join(s))                    
         
        return res.replace('#',"")
    except:
        return text
        
    

