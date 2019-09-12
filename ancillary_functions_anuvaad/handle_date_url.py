import re
import ancillary_functions_anuvaad.common_util_functions as common_utils


def tag_number_date_url(text):
  try: 
    resultant_str = list()
    count_date = 0
    date_original = list()
    count_url = 0
    url_original = list()
    for word in text.split():
        print("word",word)
        # if len(word)>4 and len(word)<12 and token_is_date(word):
        ext = [".",",","?","!"]
        if word.isalpha()== False and word[:-1].isalpha() == False and len(word)>4 and common_utils.token_is_date(word):
            if word.endswith(tuple(ext)):
              end_token = word[-1]
              word = word[:-1]
              if len(word)<7 and int(word):
                word = word+end_token
                print("kkkk")
              else:
                date_original.append(word)
                word = 'DdAaTtEe'+str(count_date)+end_token
                count_date +=1
                print("jjj")
            else:
              date_original.append(word)  
              word = 'DdAaTtEe'+str(count_date)
              count_date +=1
              print("ggg")
        elif common_utils.token_is_url(word):
            url_original.append(word)
            word = 'UuRrLl'+str(count_url)
            count_url +=1
            print("kkk")

        resultant_str.append(word)   
        s = [str(i) for i in resultant_str] 
        res = str(" ".join(s))  
    print("res",res,date_original,url_original)    

    return res,date_original,url_original 
  except Exception as e:
    print(e)   

def replace_tags_with_original(text,date_original,url_original):
  try:
    resultant_str = list()
    for word in text.split():
      print("word-1",word[:-1])
      if word[:-1] == 'DdAaTtEe':
        word = date_original[int(word[-1])]
        print(word,"date")
      elif word[:-1] == 'UuRrLl':
        word = url_original[int(word[-1])]  
        print("url",word)

      resultant_str.append(word)
      s = [str(i) for i in resultant_str] 
      res = str(" ".join(s))

    print(res,"response")
    return res    
  except Exception as e:
    print(e)
    pass

# res,date_original,url_original = tag_number_date_url('Precedent Issue 81 July/August 2007. Available at http://www.austlii.edu.au/au/journals/PrecedentAULA/2007/66.pdf.')
# replace_tags_with_original('प्राथमिक अंक DdAaTtEe0 जुलाई / अगस्त DdAaTtEe1 UuRrLl0 Ut उपलब्ध है',date_original,url_original)
"merge below two functions and above two, when training for tamil again..above two are used in tamil 2108, rest all will use below one"

def tag_number_date_url_1(text):
  try: 
    resultant_str = list()
    count_date = 0
    date_original = list()
    count_url = 0
    url_original = list()
    count_number = 0
    # number_original = list()

    hindi_numbers = ['०', '१', '२', '३','४','५','६','७','८','९','१०','११','१२','१३','१४','१५','	१६','	१७','	१८','१९','२०','२१','२२','२३','२४','२५','२६','२७','२८','२९','३०']

    num_array = re.findall(r'\d+',text) 
    # print("num_arr",num_array)  
    num_array = list(map(int, num_array)) 
    num_array.sort(reverse = True)
    for j in num_array:
      text = text.replace(str(j),'NnUuMm'+str(hindi_numbers[count_number]),1)
      count_number +=1
      if count_number >30:
        print("count exceeding 30")
        count_number = 30

    print("number tagging done")
    for word in text.split():
        print("word",word)
        # if len(word)>4 and len(word)<12 and token_is_date(word):
        ext = [".",",","?","!"]
        if word.isalpha()== False and word[:-1].isalpha() == False and len(word)>4 and common_utils.token_is_date(word):
            if word.endswith(tuple(ext)):
              end_token = word[-1]
              word = word[:-1]
              if len(word)<7 and int(word):
                word = word+end_token
                print("kkkk")
              else:
                date_original.append(word)
                word = 'DdAaTtEe'+str(count_date)+end_token
                count_date +=1
                print("jjj")
            else:
              date_original.append(word)  
              word = 'DdAaTtEe'+str(count_date)
              count_date +=1
              print("ggg")
        elif common_utils.token_is_url(word):
            url_original.append(word)
            word = 'UuRrLl'+str(count_url)
            count_url +=1
            print("kkk")

        resultant_str.append(word)   
        s = [str(i) for i in resultant_str] 
        res = str(" ".join(s))  
    print("res",res,date_original,url_original,num_array)    

    return res,date_original,url_original,num_array 
  except Exception as e:
    print(e)   

def replace_tags_with_original_1(text,date_original,url_original,num_array):
  try:
    resultant_str = list()
    hindi_numbers = ['०', '१', '२', '३','४','५','६','७','८','९','१०','११','१२','१३','१४','१५','	१६','	१७','	१८','१९','२०','२१','२२','२३','२४','२५','२६','२७','२८','२९','३०']
    array = re.findall(r'NnUuMm.', text)  
    print("NnUuMm array after translation",array) 
    for j in array:
      end_hin_number = j[-1]
      index = hindi_numbers.index(end_hin_number)
      text = text.replace(j,str(num_array[index]),1)

    for word in text.split():
      if word[:-1] == 'DdAaTtEe':
        word = date_original[int(word[-1])]
        print(word,"date")
      elif word[:-1] == 'UuRrLl':
        word = url_original[int(word[-1])]  
        print("url",word)        

      resultant_str.append(word)
      s = [str(i) for i in resultant_str] 
      res = str(" ".join(s))

    print(res,"response")
    return res    
  except Exception as e:
    print(e)
    pass
