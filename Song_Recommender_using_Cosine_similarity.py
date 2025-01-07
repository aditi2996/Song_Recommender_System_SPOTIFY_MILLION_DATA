#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("spotify_millsongdata.csv")


# In[3]:


df.shape


# In[4]:


df.isnull().sum()


# In[5]:


df.head(10)


# In[8]:


df = df.sample(10000)


# In[9]:


df = df.drop('link', axis = 1)


# In[10]:


df['text'] = df['text'].str.lower().replace(r'\r','', regex = True).replace(r'\n', '', regex = True).replace(r'^\w\s',' ', regex = True)


# In[11]:


import nltk
from nltk.stem.porter import PorterStemmer
stemer = PorterStemmer()

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stem = [stemer.stem(w) for w in tokens]
    
    return " ".join(stem)


# In[12]:


tokenization("you are beautiful, beauty")


# In[13]:


nltk.download('punkt')


# In[14]:


df['text'] = df['text'].apply(lambda x: tokenization(x))


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[16]:


tfidvector = TfidfVectorizer(analyzer = 'word', stop_words= 'english')
matrix = tfidvector.fit_transform(df['text'])

similarity = cosine_similarity(matrix)


# In[17]:


similarity[1]


# In[19]:


def recommendation(song_name):
    idx = df[df['song'] == song_name].index[0]
    distance = sorted(list(enumerate(similarity[idx])), reverse = True, key = lambda x:x[1])
    
    songs = []
    for id in distance[1:21]:
        songs.append(df.iloc[id[0]].song)
    return songs


# In[20]:


recommendation('Hello Old Friend')


# In[ ]:




