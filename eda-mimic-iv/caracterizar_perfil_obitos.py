#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 10:15:52 2023

@author: brunobmp
"""

#!/usr/bin/env python3
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

testYears = ['2017 - 2019', '2014 - 2016', '2011 - 2013', '2008 - 2010']
models = {}
## Create embeddings of ICD codes and the special ['DEATH'] token over time
for yearInterval in tqdm(testYears):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    deaths = view[~view.dod.isna()].copy()
    icds = deaths.sort_values(by=["subject_id", "seq_num"]).groupby(by=["subject_id"]).long_title.apply(list)
    icds = icds.apply(lambda x: x + ["DEATH"])
    model = Word2Vec(icds.values, vector_size=100)
    models.update({yearInterval:model})

allDiseases = patientsWithIcds.long_title.unique().tolist() + ["DEATH"]

similarityMatrix = {}
for m in tqdm(models):
    allVectors = [models[m].wv[x] for x in models[m].wv.index_to_key]
    distances = 1 - cosine_distances(allVectors, allVectors)
    dfDistances = pd.DataFrame(distances, index=models[m].wv.index_to_key, columns=models[m].wv.index_to_key)
    death = dfDistances["DEATH"]
    similarityMatrix.update({m:death})
    
    
df = pd.DataFrame(most_similar)
deathChanges = df.dropna().head(10)

plt.figure(figsize=(14,8))
plt.title("Context change over time for the special 'DEATH' token")
deathChanges.T.plot.line(ax=plt.gca())