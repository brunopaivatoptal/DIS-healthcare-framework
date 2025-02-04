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
from Utils import *
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

testYears = ['2017 - 2019', '2014 - 2016', '2011 - 2013', '2008 - 2010']
models = {}
## Create embeddings of ICD codes and the special ['DEATH'] token over time
for yearInterval in tqdm(testYears):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    deaths = view[~view.dod.isna()].copy()
    icds = deaths.sort_values(by=["subject_id", "seq_num"]).groupby(by=["subject_id"]).icd_code.apply(list)
    icds = [[classifyICD_Code(y) for y in x] + ["DEATH"] for x in tqdm(icds.values)]
    model = Word2Vec(icds, vector_size=100)
    models.update({yearInterval:model})

allDiseases = patientsWithIcds.long_title.unique().tolist() + ["DEATH"]

similarityMatrix = {}
for m in tqdm(models):
    allVectors = [models[m].wv[x] for x in models[m].wv.index_to_key]
    distances = 1 - cosine_distances(allVectors, allVectors)
    dfDistances = pd.DataFrame(distances, index=models[m].wv.index_to_key, columns=models[m].wv.index_to_key)
    death = dfDistances["DEATH"]
    similarityMatrix.update({m:death})
    
    
df = pd.DataFrame(similarityMatrix)
deathChanges = df.dropna()
diff = deathChanges["2017 - 2019"] - deathChanges["2008 - 2010"]
diff = diff.sort_values()
diff = diff[diff.index != "N/A"].copy()

K=5
plt.figure(figsize=(12,6))
plt.suptitle("Context change over time for the special 'DEATH' token from 2008-2010 to 2017-2019")
plt.subplot(2,1,1)
plt.title(f"Top-{K} that became less similar")
diff.head(K).plot.barh()
plt.subplot(2,1,2)
plt.title(f"Top-{K} that became more similar")
diff.tail(K).plot.barh()
plt.tight_layout()



## Validation
patientsWithIcds["IsCancer"] = patientsWithIcds.icd_code.str.contains("C|D")
patientsWithIcds["IsDead"] = ~patientsWithIcds.dod.isna()

patients = pd.concat([
    patientsWithIcds.groupby(by=["subject_id", "anchor_year_group"]).IsCancer.max().rename("IsCancer"),
    patientsWithIcds.groupby(by=["subject_id", "anchor_year_group"]).IsDead.max().rename("death")
    ], axis=1).reset_index()


cancer = patients.groupby(by=["anchor_year_group"]).IsCancer.sum()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Number of patients with Cancer over time")
cancer.plot.barh()
plt.xlabel("Frequency")
plt.ylabel("Anchor Year Group")

cancer = patients[patients.IsCancer == True].copy()
cancerLeathality = cancer.groupby(by=["anchor_year_group"]).death.mean()

plt.subplot(1,2,2)
plt.title("Fraction of patients with Cancer causing death over time")
cancerLeathality.plot.barh()
plt.xlabel("Frequency")
plt.ylabel("")
plt.tight_layout()



testYears = ['2017 - 2019', '2014 - 2016', '2011 - 2013', '2008 - 2010']
models = {}
## Create embeddings of ICD codes and the special ['DEATH'] token over time
for yearInterval in tqdm(testYears):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    deaths = view[~view.dod.isna()].copy()
    icds = deaths.sort_values(by=["subject_id", "seq_num"]).groupby(by=["subject_id"]).icd_code.apply(list)
    icds = [[y[:3] for y in x if classifyICD_Code(y) == "Neoplasms"] + ["DEATH"] for x in tqdm(icds.values)]
    model = Word2Vec(icds, vector_size=100)
    models.update({yearInterval:model})

allDiseases = patientsWithIcds.long_title.unique().tolist() + ["DEATH"]

similarityMatrix = {}
for m in tqdm(models):
    allVectors = [models[m].wv[x] for x in models[m].wv.index_to_key]
    distances = 1 - cosine_distances(allVectors, allVectors)
    dfDistances = pd.DataFrame(distances, index=models[m].wv.index_to_key, columns=models[m].wv.index_to_key)
    death = dfDistances["DEATH"]
    similarityMatrix.update({m:death})
    

codes = pd.read_csv("icd-codes.csv", header=None)
codes[0] = codes[0].apply(lambda x: x[:3])
codes = codes.set_index(0)[5].to_dict()

df = pd.DataFrame(similarityMatrix)
df.index  = [codes[x] if x in codes else x for x in df.index]
deathChanges = df.dropna()
diff = deathChanges["2017 - 2019"] - deathChanges["2008 - 2010"]
diff = diff.sort_values()
diff = diff[diff.index != "N/A"].copy()

K=8
plt.figure(figsize=(12,6))
plt.suptitle("Context change over time for the special 'DEATH' token from 2008-2010 to 2017-2019")
plt.subplot(2,1,1)
plt.title(f"Top-{K} that became less similar with the 'DEATH' token")
diff.head(K).plot.barh()
plt.subplot(2,1,2)
plt.title(f"Top-{K} that became more similar with the 'DEATH' token")
diff.tail(K).plot.barh()
plt.tight_layout()