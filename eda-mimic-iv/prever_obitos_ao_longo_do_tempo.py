# -*- coding: utf-8 -*-
"""
Caracterização do MIMIC-IV na tarefa de predição de óbito usando doenças de base.
"""

from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
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

root = [x for x in [
    Path("../../DATASETS/MIMIC-IV/mimic-iv-2.2"),
    Path("../../../DATASETS/MIMIC-IV/mimic-iv-2.2")]
    if os.path.exists(x)][0]

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
patientsWithLab = pd.merge(patients, labEvents.loc[:,["subject_id","itemid"]], on=["subject_id"])

clf=None
metrics = []

## baseline 
patientsWithIcds["obito"] = (~patientsWithIcds.dod.isna()).astype(int)
X_baseline_raw = patientsWithIcds.groupby(by=["subject_id"]).icd_code.apply(list).apply(lambda x: " ".join(x))
y_baseline = patientsWithIcds.groupby(by=["subject_id"]).obito.max()

X_train, X_test, y_train, y_test = train_test_split(X_baseline_raw, y_baseline, test_size=0.25)
baselineVectorizer = TfidfVectorizer()
X_train_baseline_vect = baselineVectorizer.fit_transform(X_train)
X_test_baseline_vect = baselineVectorizer.transform(X_test)
clfBaseline = lgb.LGBMClassifier().fit(X_train_baseline_vect, y_train)
y_pred = clfBaseline.predict(X_test_baseline_vect)

metrics.append([{
    "macroF1":f1_score(y_test, y_pred, average="macro"),
    "microF1":f1_score(y_test, y_pred, average="micro"),
    "recall":recall_score(y_test, y_pred),
    "classifier":"All time-slices"
    }])



for yearInterval in tqdm(patientsWithIcds.anchor_year_group.unique()):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    view["obito"] = (~view.dod.isna()).astype(int)
    gb = view.groupby(by=["subject_id"])
    X_raw = gb.icd_code.apply(list).apply(lambda x: " ".join(x))
    y_raw = gb.obito.max()
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw)
    
    if clf is None:
        vectorizer = TfidfVectorizer()
        X_train_vect = vectorizer.fit_transform(X_train)
        clf = lgb.LGBMClassifier().fit(X_train_vect, y_train)
        
    X_test_vect = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vect)
    
    metrics.append([{
        "macroF1":f1_score(y_test, y_pred, average="macro"),
        "microF1":f1_score(y_test, y_pred, average="micro"),
        "recall":recall_score(y_test, y_pred),
        "classifier":"First time slice",
        "year":yearInterval
        }])
        
    partitionVectorizer = TfidfVectorizer()
    X_train_vect = partitionVectorizer.fit_transform(X_train)
    partitionClf = lgb.LGBMClassifier().fit(X_train_vect, y_train)
    X_test_vect = partitionVectorizer.transform(X_test)
    y_pred = partitionClf.predict(X_test_vect)
    
    metrics.append([{
        "macroF1":f1_score(y_test, y_pred, average="macro"),
        "microF1":f1_score(y_test, y_pred, average="micro"),
        "recall":recall_score(y_test, y_pred),
        "classifier":"Same time slice",
        "year":yearInterval
        }])



dfAcc = pd.concat([pd.Series(m[0]) for m in metrics], axis=1).T.sort_index()
print(dfAcc.to_string())

dfAcc.drop(columns=["year"]).groupby(by=["classifier"]).mean()


clfItems = None
metricsItems = {}

for yearInterval in tqdm(patientsWithLab.anchor_year_group.unique()):
    view = patientsWithLab[patientsWithLab.anchor_year_group == yearInterval].copy()
    view["obito"] = (~view.dod.isna()).astype(int)
    gb = view.groupby(by=["subject_id"])
    X_raw = gb.itemid.apply(list).apply(lambda x: " ".join([str(y) for y in x]))
    y_raw = gb.obito.max()
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw)
    
    if clfItems is None:
        vectorizerItems = TfidfVectorizer()
        X_train_vect = vectorizerItems.fit_transform(X_train)
        clfItems = lgb.LGBMClassifier().fit(X_train_vect, y_train)
        
    X_test_vect = vectorizerItems.transform(X_test)
    y_pred = clfItems.predict(X_test_vect)
    
    metricsItems.update({yearInterval:{
        "macroF1":f1_score(y_test, y_pred, average="macro"),
        "microF1":f1_score(y_test, y_pred, average="micro"),
        "recall":recall_score(y_test, y_pred)
        }})


dfAccItems = pd.DataFrame(metricsItems).T.sort_index()


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plt.title("Classification metrics of death prediction using diseases")
dfAcc.plot.line(ax=plt.gca())

plt.subplot(1,2,2)
plt.title("Classification metrics of death prediction using lab items")
dfAccItems.loc[dfAcc.index,:].plot.line(ax=plt.gca())