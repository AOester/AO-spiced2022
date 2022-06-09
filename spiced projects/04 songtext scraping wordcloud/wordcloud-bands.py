import imageio

import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import accuracy_score

from wordcloud import WordCloud as WC
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import seaborn as sns

sns.axes_style('darkgrid')

group1 = 'Genesis'
group2 = 'Soundgarden'


def readsongtitles(Band,longorshort='short'):
    
    List = []
    if longorshort == 'short':
        print('reading short list')
        with open (f"web_scraping_cf/DataMusic/{Band}_list.txt",'r') as file:
            List = file.read()
            liste = List.splitlines()
    elif longorshort == 'long':
        print('reading long list')
        with open (f"web_scraping_cf/DataMusic/{Band}_list_long.txt",'r') as file:
            List = file.read()
            liste = List.splitlines()
    else :
        liste = []
        return f"not valid, only 'long' / 'short'"
    
    return liste

def grabcontent(grouplist): # from file
    content = []
    print(f"start reading files")
    for i in grouplist:
        print(f"reading lyrics: {i}")
        with open (f"web_scraping_cf/DataMusic/{i}.txt",'r') as g:
            content.append(' '.join(g.read().split()))
    print(f"end reading files")
    return content


group1list = readsongtitles(group1,'short')
group2list =  readsongtitles(group2,'short')

group1content = grabcontent(group1list)
group2content = grabcontent(group2list)


#group1content = grabcontent(group1list)
#group2content = grabcontent(group2list)

#print(group2)
#print(group2content)

genesiswords = " ".join(group1content)
soundgardenwords = " ".join(group2content)


# Generate a word cloud image
genesiscloud = WC(background_color="white",width=1440, height=900).generate(genesiswords)
soundgardencloud = WC(background_color="white",width=1440, height=900).generate(soundgardenwords)

plt.figure(figsize=(14.4, 9))

#plt.subplot(2,1,1)
plt.title('Wordcloud Genesis', fontdict = {'fontsize' : 20})
plt.imshow(genesiscloud, interpolation="bilinear")
#plt.grid(None)
plt.axis('off')
plt.savefig('Genesis.png',dpi=128,bbox_inches='tight')

#plt.show()

#plt.figure(figsize=(14.4, 9))
#plt.subplot(2,1,2)
plt.title('Wordcloud Soundgarden', fontdict = {'fontsize' : 20})
plt.imshow(soundgardencloud, interpolation="bilinear")
#plt.grid(None)
plt.axis('off')
plt.savefig('Soundgarden.png', dpi = 128,bbox_inches='tight')

#plt.show()
