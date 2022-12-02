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
import tensorflow

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

df = pd.DataFrame(columns = cols)
# df = pd.read_pickle(r'what_mps_say.pickle')

start_date = "2022-01-01"
end_date = "2022-07-01"
# date format is y-d-m
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
        

#%%

# from transformers import TFDistilBertTokenizer, TFDistilBertForSequenceClassification

# tokenizer = TFDistilBertTokenizer.from_pretrained("distilbert-base-uncased")
# model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits

# predicted_class_id = logits.argmax().item()
# model.config.id2label[predicted_class_id]
