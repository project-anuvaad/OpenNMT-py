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
