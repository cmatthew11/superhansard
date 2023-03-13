# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:30:55 2022

@author: Chris
"""

from bs4 import BeautifulSoup
import pandas as pd
import requests
import re
import numpy as np
from transformers import pipeline as pipe
# import tensorflow
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import IPython
from tqdm import tqdm
import umap
import hdbscan

def strip(string):
    """removes whitespace and newlines from text"""
    out = string.replace("\n","")
    out = re.sub(' +', ' ',out)
    out = out.lower()
    return out

def get_soup(url):
    rall = requests.get(url)
    r = rall.content
    soup = BeautifulSoup(r,"lxml")
    return soup

def match_names(string):
    
    """returns all the indexes from the mps df whose names are in the string"""
    out = []
    for i in mps.name:
        if i in string:
            out.append(name2ind[i])
    for j in mps.position:
        if j in string:
            ind = mps.loc[mps.position==j].index
            for k in ind:
                if k not in out:
                    out.append(k)
    return out

def get_speech(soup): 
    """takes soup of a debate page and returns who said what in a df"""
    header = soup.h3.text
    divs = soup.find_all("div",{"class":"col-sm-12 mt-2 mb-2"})
    if len(divs)==0:
        return pd.DataFrame()
    out = []
    for div in divs:
        if bool(div.a):
            mp_ind = match_names(strip(div.a.text))
            if len(mp_ind) > 0:
                # continue
                speech = div.find("p").text
                out.append({"mp_ind":mp_ind[0],
                            "speech":speech,
                            "debate":header,
                            "name":ind2name[mp_ind[0]]})
    return pd.DataFrame(out)

def convert_date(date):
    """gets date in the format needed for url"""
    day = str(date.day)
    month = str(date.month)
    year = str(date.year)
    date = year+"-"+month+"-"+day
    return date


mps = pd.read_csv('mps- parallelparliament.csv',
                  index_col = 0)

mps["name"] = mps.Name.apply(lambda q: q.lower())
mps['position'] = mps.Position.apply(lambda q: str(q).lower())

""""ignoring the pm here cause it changes more interested in general vibe"""
mps = mps.loc[~mps.position.str.contains("prime minister")]
name2ind = dict(zip(mps.name,mps.index))
ind2name = dict(zip(mps.index,mps.name))

base_url = 'https://www.parallelparliament.co.uk'
cols = ["mp_ind","date","speech","debate","name"]


#%%

# df = pd.DataFrame(columns = cols)
df = pd.read_pickle(r'what_mps_say.pickle')

start_date = "2022-01-01"
end_date = "2022-07-01"
# date format is y-d-m
if len(df)==0:
        
    for date in pd.date_range(start_date,end_date):
        # if date.day==1:
        #     df.to_pickle("what_mps_say.pickle")
        date = convert_date(date)
        if date not in df.date:
            print(date)
            url = base_url + '/debate/' + date
            soup = get_soup(url)
            urls = [base_url + i["href"] for i in soup.find_all("a",text="Read Full debate")]
            for url in urls:
                soup = get_soup(url)
                speech = get_speech(soup)
                speech["date"] = date
                if speech.shape[0] > 0:
                    print(speech.debate[0])
                df = pd.concat([df,speech])
        
#%%

df["char_count"] = df.speech.apply(lambda q: len(q))
df["speech_strip"] = df.speech.apply(strip)
df["word_count"] = df.speech.apply(lambda q: q.count(" "))
df.reset_index(inplace = True)
        
#%% https://towardsdatascience.com/untangling-uk-politics-with-data-science-a5afe9a86923

N = 'en_core_web_sm'

try:
    nlp = spacy.load(N) 
except:
    spacy.cli.download(N)
    nlp = spacy.load(N)
    
is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")


# turn all the text into vectors
n = len(df)
vectorizer = TfidfVectorizer(min_df=0.0,lowercase=True)
sample = df.speech.values[:n+1]
tfidf = vectorizer.fit(sample)
tkn = tfidf.build_tokenizer()
print('creating a lookup dictionary') #this speeds up the script significantly...
tfidf_lookup = {}
for key,value in tfidf.vocabulary_.items():
   tfidf_lookup[key]=tfidf.idf_[value]
vect = []
for doc in tqdm(nlp.pipe(sample,batch_size=n)):
   weighted_doc_tensor = []
   try:
      for cnt, wrd_vec in enumerate(doc.tensor):
         word = doc[cnt].text
         try:
            weight = tfidf_lookup[word.lower()]
         except:
            weight = 0.5
         pass
         doc.tensor[cnt] = doc.tensor[cnt]*weight
      vect.append(np.mean(doc.tensor,axis=0))
   except:
      vect.append(np.zeros(768,))#if it is blank...
   pass
vect = np.vstack(vect)
# np.save('question_vects_tfidf.npy', vect)

#add in to replace nans (only happened with large df)
vect = np.nan_to_num(vect)

#reduce dimensions of hte array of vectors
reducer = umap.UMAP(n_neighbors=5)
df.loc[:n,["x_umap_5","y_umap_5"]] = reducer.fit_transform(vect)
reducer = umap.UMAP(n_neighbors=25)
df.loc[:n,["x_umap_25","y_umap_25"]] = reducer.fit_transform(vect)

#hdbscan to cluster vectors
db = hdbscan.HDBSCAN(min_cluster_size=40, min_samples=1).fit(df[["x_umap_5","y_umap_5","x_umap_25","y_umap_25"]])
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
df['cluster'] = db.labels_

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
tfidf = vectorizer.fit_transform(df.speech.values[:n])
totals = 0
for cluster in df.cluster.value_counts()[0:50].index:
   stg = " ".join(df.loc[df.cluster==cluster].speech.values[:n])
   response = vectorizer.transform([stg])
   count = df.cluster.value_counts().loc[cluster]
   totals += count
   feature_array = np.array(vectorizer.get_feature_names())
   tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
   n = 10
   print("Cluster Label: {}, Items in Cluster: {}".format(cluster,count))
   print(feature_array[tfidf_sorting][:n])

WHAT ABOUT EXCTRACTING KEYWORDS (NOUNS PLACES THINGS ETC) to compare with APPG topic keywords (maybe extracted from GOOGLE)
