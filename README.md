# Portfolio

# [Twitter Sentiment Analysis](https://github.com/nezihaksu/Projects/tree/master/medicalsentiment)
  
  Sentiment analysis yields results beyond words' meanings and their meaning in sentences.
  
  Representing words in counts,frequencies,ratios,machine learning models regarding their labels opens up new analysis ways to           
  understand underlying meaning.
  
  In this dataset there are two labels 0 and 1,negative and positive.By analyzing this dataset i aim to first of all to see what words     are being used as negative and positive,their frequency by negative and positive classes.There can be pretty shocking results.
  
  Then creating a model based on cleaned dataset to classify future tweets as negative or positive.
  
  These are the steps to analyze:
  
  -Cleaning text data
  -EDA
  -Model

# Cleaning Text Data
  Texts are scraped from the twitter api,so these texts can include hashtags,usernames,url links.
  
  
# EDA
  Introduction to dataset,visulization and understanding the words beyond their meanings and sentence meaning.
  
  Wordcloud for negative words:
  
![](/images/negative_wordcloud.JPG)
  
  Wordcloud for positive words:
  
![](/images/positive_wordcloud.JPG)

  Which word is used for what sense and their ratio of negative or positive usage.There are thousands of words,these are just the best     representations that i found.
  
![](https://github.com/nezihaksu/Projects/blob/master/images/neg-pos-ratio.jpeg)

# Model

  Searched the best model and best parameters for the best model with the gridsearch module from sklearn.
  Usually logistic regression is the best machine learning algorithm that classifies binary classes,even better than neural networks    
  mostly.
  Both timewise and simplicity logistic regression is the way to go when there are binary outcomes.
  
