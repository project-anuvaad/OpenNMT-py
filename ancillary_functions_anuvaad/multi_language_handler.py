from textblob import TextBlob
import re

'''
Currently only handling foreign words(english) during vernacular to english translation
UuRrLl token not working correctly need to retrain the model with new tokens
'''
def tag_english_words(text):
    try:
        regex = r'[A-Za-z]+'
        list1=re.findall(regex,text)
        count_word = 0

        for j in list1:
            text = text.replace(str(j),'UuRrLl'+str(count_word),1)
            count_word +=1
        print(text)
        print(list1)
        return text,list1
    except Exception as e:
        print(e)

def replace_english_words(text,word_arr):
    try:      
        if len(text) == 0:
            return ""

        word_tagged_arr = re.findall(r'UuRrLl..|UuRrLl.', text)   
        for j in word_tagged_arr:
            j = j.replace(' ','')
            if j[:-1] == "UuRrLl":
                text = text.replace(j,(word_arr[int(j[-1])]),1)
            elif j[-2:].isnumeric():
                text = text.replace(j,(word_arr[int(j[-2:])]),1) 
            else:
                text = text.replace(j[:-1],(word_arr[int(j[-2])]),1) 

        print("response after url and date replacemnt:{}".format(text))
        return text
    except Exception as e:
        print(e)