import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import re

from nltk.tokenize import WordPunctTokenizer
tokenizer = WordPunctTokenizer()

from bs4 import BeautifulSoup as bs

training_csv = r'C:\Users\nezih\Desktop\medicalsentiment.csv'

cols = ['sentiment','id','date','query_string','user','text']

df = pd.read_csv(training_csv,engine = 'python',names = cols)

df = df.drop(['id','date','query_string','user'],axis = 1)


#STRIPS HASHTAGS AND LINKS.
hashtag = r'@[A-Za-z0-9]+'
link = r'https?://[A-Za-z0-9./]+'
link0 = r'www*com'


hashtag_link = r'|'.join((hashtag,link,link0))
hashtag_links_re = re.compile(hashtag_link)

special_char_dict = {df.text[226][7]:"",df.text[226][8]:"",df.text[226][9]:""}
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}

def multiple_replace(dict,text):

	regex = re.compile("%s"%"|".join(map(re.escape,dict.keys())))

	return regex.sub(lambda mo:dict[mo.string[mo.start():mo.end()]],text)



def tweet_cleaner(text,regex):

	soup = bs(text,features = 'lxml')
	soup_text = soup.get_text()
	stripped = re.sub(regex,'',soup_text)

	try:
		
		utf8_stripped = re.sub(u"\ufffd","",stripped.decode("utf-8-sig"))

	except:

		utf8_stripped = multiple_replace(special_char_dict,stripped)



	negations = multiple_replace(negations_dic,utf8_stripped)

	letters_only = re.sub(r'[0-9]',"",negations)

	lower_case = letters_only.lower()

	each_word = [word for word in tokenizer.tokenize(lower_case) if len(word) > 1]

	return (' '.join(each_word).strip())



if __name__ == '__main__':

	clean_text = [tweet_cleaner(row,hashtag_links_re) for row in df.text]
	
	new_df = pd.DataFrame(clean_text,columns = ['text'])
	
	csv_name = 'cleaned_tweets.csv'
	
	new_df.to_csv(csv_name,encoding = 'utf-8',index = False)



	





