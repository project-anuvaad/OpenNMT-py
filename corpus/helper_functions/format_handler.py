"This is for pre-processing the traning corpus so that special characters, dates, numbers etvc. are handled separately"

import re
import sys
from dateutil.parser import parse

"below is for replacing numbers with #" "not using now"
def replace_number_with_hash(in_file,out_file):
  try:
    with open("{0}".format(in_file)) as xh:
      with open("{0}".format(out_file),"w") as zh:
        xlines = xh.readlines() 
        for i in range(len(xlines)):
        # line = re.sub(r'[^A-Za-z0-9. ]+', '$', xlines[i])
          line = re.sub(r'\d', '#', xlines[i])
          zh.write(line)
  except:
    print("replace number with hash exception pass")
    pass
  
"below is for replacing hindi numbers with corresponding english"
def replace_hindi_numbers(in_file,out_file):
  try:
    hindi_numbers = ['०', '१', '२', '३','४','५','६','७','८','९']
    eng_numbers = ['0','1','2','3','4','5','6','7','8','9']
    outfile = open("{0}".format(out_file), "w")
    for line in open("{0}".format(in_file), "r"):
      for i in hindi_numbers : 
        line = line.replace(i,eng_numbers[hindi_numbers.index(i)])
      outfile.write(line)
    outfile.close()
  except:
    print("in replace_hindi_number pass")
    pass

"below is for replacing links with tag UuRrLl using regex"   
def replace_links_with_tag(in_file,out_file):
  try:
    outfile = open("{0}".format(out_file), "w")
    for line in open("{0}".format(in_file), "r"):
        #  url = re.findall(r'http[s]?\s*:\s*/\s*/\s*(?:\s*[a-zA-Z]|[0-9]\s*|[$-_@.&+]|\s*[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]\s*))+', line)
         line = re.sub(r'http[s]?\s*:\s*/\s*/\s*(?:\s*[a-zA-Z]|[0-9]\s*|[$-_@.&+]|\s*[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]\s*))+','UuRrLl',line)
         outfile.write(line)
    outfile.close() 
  except:
    print("replace links with tag")
    pass

def tag_number_date_url(in_file,out_file):
  try:    
    with open("{0}".format(in_file)) as xh:
      with open("{0}".format(out_file),"w") as zh:
        xlines = xh.readlines() 
        for i in range(len(xlines)):
          resultant_str = list()
          count_date = 0
          count_url = 0
          for word in xlines[i].split():
            if word.isalpha()== False and len(word)>4 and word.endswith('.')==False and token_is_date(word):
              word = 'DdAaTtEe'+str(count_date)
              count_date +=1
              # print("ggg")
            elif token_is_url(word):
              word = 'UuRrLl'+str(count_url)
              count_url +=1
              # print("kkk")

            resultant_str.append(word)   
            s = [str(i) for i in resultant_str] 
            res = str(" ".join(s)) 
          zh.write(str(res))
          zh.write('\n')     

    
  except Exception as e:
    print("in error pass format_handler: ",e)
    pass
    
  

def isfloat(str):
    try: 
        float(str)
    except ValueError: 
        return False
    return True

def token_is_date(token):
    try: 
        parse(token, fuzzy=False)
        return True

    except ValueError:
        return False
    except OverflowError:
      print("overflow error while parsing date, treating them as Date tag{}".format(token))
      return True    
    except Exception as e:
      print("error in date parsing for token:{} ".format(token),e)
      return False

def token_is_url(token):
  try:
    url = re.findall(r'http[s]?\s*:\s*/\s*/\s*(?:\s*[a-zA-Z]|[0-9]\s*|[$-_@.&+]|\s*[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]\s*))+',token)
    if len(url)>0:
      return True
    else:
      return False  
  except:
    return False

