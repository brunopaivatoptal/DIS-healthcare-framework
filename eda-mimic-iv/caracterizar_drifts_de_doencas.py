#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 08:55:39 2023

@author: brunobmp
"""

# -*- coding: utf-8 -*-
"""
Caracterização do MIMIC-IV na tarefa de predição de óbito usando doenças de base.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from pathlib import Path
import lightgbm as lgb
from tqdm import tqdm
import pandas as pd
import numpy as np
import sys
import os
import re


if sys.platform.find("darwin")>=0 :
    root = Path("../../DATASETS/MIMIC-IV/mimic-iv-2.2")
elif sys.platform.find("win") >= 0 :
    root = Path("../../../DATASETS/MIMIC-IV/mimic-iv-2.2")
else:
    print("Platform not supported !")


patients = pd.read_csv(root/"hosp"/"patients.csv.gz")
"""
patients.columns
Index(['subject_id', 'gender', 'anchor_age', 'anchor_year',
       'anchor_year_group', 'dod'],
      dtype='object')
"""

icdSubject = pd.read_csv(root/"hosp"/"diagnoses_icd.csv.gz")
icdDiagn = pd.read_csv(root/"hosp"/"d_icd_diagnoses.csv.gz")
labEvents =  pd.read_csv(root/"hosp"/"labevents.csv.gz")
"""
labEvents.columns
Index(['labevent_id', 'subject_id', 'hadm_id', 'specimen_id', 'itemid',
       'order_provider_id', 'charttime', 'storetime', 'value', 'valuenum',
       'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority',
       'comments'],
      dtype='object')
"""
icdDiagn = icdDiagn[icdDiagn.icd_version == 10].copy()

icds = pd.merge(icdSubject, icdDiagn, on=["icd_code", "icd_version"]).sort_values(by=["subject_id","seq_num"])


patientsWithIcds = pd.merge(patients, icds, on=["subject_id"])
"""
patientsWithIcds.columns
Index(['subject_id', 'gender', 'anchor_age', 'anchor_year',
       'anchor_year_group', 'dod', 'hadm_id', 'seq_num', 'icd_code',
       'icd_version', 'long_title'],
      dtype='object')
"""

vectorizer=None
metrics = {}

for yearInterval in tqdm(sorted(patientsWithIcds.anchor_year_group.unique())):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    view["obito"] = (~view.dod.isna()).astype(int)
    gb = view.groupby(by=["subject_id"])
    X_raw = gb.icd_code.apply(list).apply(lambda x: " ".join(x))
    y_raw = gb.obito.max()
    
    if vectorizer is None:
        vectorizer = CountVectorizer()
        refX = vectorizer.fit_transform(X_raw).mean(axis=0)
        
    X_vect = vectorizer.transform(X_raw).mean(axis=0)
    cosineDist = cosine_distances(refX, X_vect)
    
    metrics.update({yearInterval:{
        "CosineDistance":cosineDist.reshape(-1),
        }})
    

dfDist = pd.DataFrame(metrics).T.sort_index()

dfDistMean = dfDist.CosineDistance.apply(lambda x: np.mean(x))


plt.figure(figsize=(8,5))
plt.title("Cosine distances of disease distributions over time")
dfDistMean.plot.line(ax=plt.gca())



vectorizer=None
metricsDeaths = {}
for yearInterval in tqdm(sorted(patientsWithIcds.anchor_year_group.unique())):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    view["obito"] = (~view.dod.isna()).astype(int)
    gb = view.groupby(by=["subject_id"])
    X_raw = gb.icd_code.apply(list).apply(lambda x: " ".join(x))
    y_raw = gb.obito.max()
    
    obitos = y_raw[y_raw == 1].index
    X_raw = X_raw.loc[obitos]
    
    if vectorizer is None:
        vectorizer = CountVectorizer()
        refX = vectorizer.fit_transform(X_raw).mean(axis=0)
        
    X_vect = vectorizer.transform(X_raw).mean(axis=0)
    cosineDist = cosine_distances(refX, X_vect)
    
    metricsDeaths.update({yearInterval:{
        "CosineDistance":cosineDist.reshape(-1),
        }})
    

dfDistDeaths = pd.DataFrame(metricsDeaths).T.sort_index().CosineDistance.apply(lambda x: np.mean(x))



plt.figure(figsize=(8,5))
plt.title("Cosine distances of disease distributions over time (Only patients that died)")
dfDistDeaths.plot.line(ax=plt.gca())



plt.figure(figsize=(10,6))
plt.title("Cosine distances of disease distributions over time (Deaths x Overall)")
dfDistMean.plot.line(ax=plt.gca(), label="Overall")
dfDistDeaths.plot.line(ax=plt.gca(), label="Deaths")
plt.legend()
