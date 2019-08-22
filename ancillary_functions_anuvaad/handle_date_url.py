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
        if word.isalpha()== False and common_utils.token_is_date(word):
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
