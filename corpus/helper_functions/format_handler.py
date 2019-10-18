"This is for pre-processing the traning corpus so that special characters, dates, numbers etvc. are handled separately"
"handling numbers like: "
"1. 1797/2013"
"2.26603-26605/2018" " NUM0-NUM1/NUM2  "
"3.26603-26605"
"4.nos.26603-26605"
"5. /166232014"
"6. 7th---NnUuMm0th"
"7. 11,11,11 --numbers like this can be treated as one nmber or 3 different numbers. we are treating three different numbers"
"Note: applying reverse sorting on array, so that largest number is num0 always"
"handling only upto 30 numbers in a string and using hindi numerals for tagging numbers"

import re
import sys
from dateutil.parser import parse
import random

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
        ext_date = list()
        ext_url = list()
        ext_num = list()
        hindi_numbers = ['०', '१', '२', '३','४','५','६','७','८','९','१०','११','१२','१३','१४','१५','	१६','	१७','	१८','१९','२०','२१','२२','२३','२४','२५','२६','२७','२८','२९','३०']
        for i in range(len(xlines)):
          resultant_str = list()
          count_date = 0
          count_url = 0
          count_number = 0
          num_array = re.findall(r'\d+',xlines[i]) 
          # print("num_arr",num_array)  
          num_array = list(map(int, num_array))  
          num_array.sort(reverse = True)
          for j in num_array:
            xlines[i] = xlines[i].replace(str(j),'NnUuMm'+str(hindi_numbers[count_number]),1)
            count_number +=1
            if count_number >30:
              print("count exceeding 30")
              count_number = 30

          for word in xlines[i].split():
            ext = [".",",","?","!"]
            if word.isalpha()== False and word[:-1].isalpha() == False and len(word)>4 and token_is_date(word):
              if word.endswith(tuple(ext)):
                print("ffff",word)
                end_token = word[-1]
                word = word[:-1]
                print("worddd",word)
                if len(word)<7 and (word):    
                  word = word+end_token
                else:
                  ext_date.append(word)
                  word = 'DdAaTtEe'+str(count_date)+end_token
                  count_date +=1
              else:
                ext_date.append(word)  
                word = 'DdAaTtEe'+str(count_date)
                count_date +=1
            elif token_is_url(word):
              ext_url.append(word)
              word = 'UuRrLl'+str(count_url)
              count_url +=1
              # print("kkk")
            # elif isfloat(word):
            #   ext_num.append(word)
            #   word = 'NnUuMm'+str(count_number) 
            #   count_number +=1

            resultant_str.append(word)   
            s = [str(i) for i in resultant_str] 
            res = str(" ".join(s)) 
          zh.write(str(res))
          zh.write('\n')  
        print("dates: ",ext_date)
        print("url: ",ext_url)
        # print("numbers: ",ext_num)     

    
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

def token_with_numbers(token):
  try:
    count_number = 0
    array = re.findall(r'\d+', token) 
    print("array unsorted",array)
    array = list(map(int, array))
    array.sort(reverse = True)
    print(array)
    for i in array:
      # token = re.sub(i,'NnUuMm'+str(count_number),token)
      hindi_numbers = ['०', '१', '२', '३','४','५','६','७','८','९','१०','११','१२','१३','१४','१५','	१६','	१७','	१८','१९','	२०']
      token = token.replace(str(i),'NnUuMm'+str(hindi_numbers[count_number]),1)
      print(token)
      # print("countnumber",count_number)
      count_number +=1
      
  except Exception as e:
    print(e)
    return False

def shuffle_file(in_file,out_file):
  "for shuffling a single file. Current use case is to shuffle tab sep file in scripts"
  try:
    lines = open(in_file).readlines()
    random.shuffle(lines)
    open(out_file, 'w').writelines(lines)
    print("shuffling successful")
    
  except Exception as e:
    print("Error: while shuffling file,in format_handler",e)


# tag_number_date_url('corpus/original_data/SC_specific_eng_020819.txt','corpus/test3.txt')

# token_with_numbers("जो संख्या किसी भी संख्या के समान भाग करती है उसे गुणन खंड कहेते है. उदाहरण स्वरूप 6, इस संख्या के 1,2,3 और 6 यह गुणन खंड है. 4 यह संख्या 6 का गुणन खंड नही है क्यांेकी 6 यह संख्या 4 समान भांगो में नही बॉट सकते. आप गुणक की तुलना किसी भी परिवार से और गुणनखंड की तुलना उस परिवार के व्यक्ती से कर सकते है. इसलिये 1,2,3 और 6 यह 6 इस परिवार के सदस्य है लेकिन 4 यह संख्या दुसरे परिवार की सदस्य. बोर्डपर घुमने के लिये कि-बोर्डके ऍरो किज्का उपयोग करे. ट्र्ॉगल्स से बचे. नंबर खाने के लिये स्पेस बार दबाये. ")
                    # जो संख्या किसी भी संख्या के समान भाग करती है उसे गुणन खंड कहेते है. उदाहरण स्वरूप NnUuMm0, इस संख्या के NnUuMmNnUuMm143,NnUuMm11,NnUuMm9 और NnUuMm1 यह गुणन खंड है. NnUuMm6 यह संख्या NnUuMmNnUuMm12 का गुणन खंड नही है क्यांेकी NnUuMmNnUuMm10 यह संख्या NnUuMm7 समान भांगो में नही बॉट सकते. आप गुणक की तुलना किसी भी परिवार से और गुणनखंड की तुलना उस परिवार के व्यक्ती से कर सकते है. इसलिये 1,2,3 और NnUuMmNnUuMm8 यह NnUuMm5 इस परिवार के सदस्य है लेकिन 4 यह संख्या दुसरे परिवार की सदस्य. बोर्डपर घुमने के लिये कि-बोर्डके ऍरो किज्का उपयोग करे. ट्र्ॉगल्स से बचे. नंबर खाने के लिये स्पेस बार दबाये.
# token_with_numbers("1,2,3,4,5,1,1")                    