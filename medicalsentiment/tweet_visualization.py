import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from wordcloud import WordCloud

tf_df = pd.read_csv('term_freq0.csv',engine = 'python')

tf_df.columns = ["features","positive","negative","total"]

tf_df.index = tf_df[tf_df.columns[0]]

tf_df.drop(tf_df.columns[0],axis = 1,inplace = True)

sorted_tf_df = tf_df.sort_values(by = 'total',ascending = False)

neg_string = [string for string in sorted_tf_df.negative]

neg_string = pd.Series(neg_string).astype(str).str

wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()



pos_tweets = df[df.sentiment == 4].text
neg_tweets = df[df.sentiment == 0].text

pos_string = [string for string in pos_tweets.text]
pos_string = pd.Series(pos_string).astype(str).str.cat(sep = ' ')

wordcloud = WordCloud(width = 1200,height = 720,max_font_size = 200).generate(pos_string)

plt.figure(figsize = (12,9))
plt.imshow(wordcloud,interpolation = "bilinear")
plt.axis("off")
plt.show()


start = 240
end = 260
features = sorted_tf_df.index[start:end]
neg_freq = sorted_tf_df.negative[start:end]
pos_freq = sorted_tf_df.positive[start:end]
total_freq = sorted_tf_df.total[start:end]
neg_ratio = round(neg_freq/(total_freq),4)
pos_ratio = round(pos_freq/(total_freq),4)

x = np.arange(len(features))

width = 0

fig,ax = plt.subplots()

rects1 = ax.bar(x - width/2,neg_ratio,label = 'Used as Negative')
rects2 = ax.bar(x + width/2,pos_ratio,label = 'Used as Positive')

ax.set_ylabel('Frequencies')
ax.set_title('Frequencies by negative and positive')
ax.set_xticks(x)
ax.set_xticklabels(features)
ax.legend()

def autolabel(rects):

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()



