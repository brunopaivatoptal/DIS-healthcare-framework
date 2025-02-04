#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 18:23:16 2023

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
icds_to_exclude = ["Z66", "Z515"]
patientsWithIcds = patientsWithIcds[~patientsWithIcds.icd_code.isin(icds_to_exclude)].copy()
"""
patientsWithIcds.columns
Index(['subject_id', 'gender', 'anchor_age', 'anchor_year',
       'anchor_year_group', 'dod', 'hadm_id', 'seq_num', 'icd_code',
       'icd_version', 'long_title'],
      dtype='object')
"""

testYears = ['2017 - 2019', '2014 - 2016', '2011 - 2013', '2008 - 2010']
correlations = []
importances = []
## Create embeddings of ICD codes and the special ['DEATH'] token over time
for yearInterval in tqdm(testYears):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    view["icd_chapter"] = [classifyICD_Code(icd) for icd in tqdm(view.icd_code)]
    view["icd_chapter"] = view["icd_chapter"].str.replace("[^A-Za-z\s]", "")
    view["dead"] = ~view.dod.isna()
    view = view[view.icd_chapter != "N/A"]
    icds = pd.pivot_table(view, index="subject_id", columns="icd_chapter", values="seq_num", aggfunc=len)
    deaths = (view.groupby(by="subject_id").dead.max()).to_dict()
    icds["death"] = [deaths[i] for i in icds.index]
    
    corr = icds.corr(method="pearson").death.dropna().sort_values()
    corr = corr[~corr.index.isin(["N/A", "death"])].rename(yearInterval)
    correlations.append(corr)
    
    X, y = icds.drop(columns=["death"]), icds.death
    X.columns = [re.sub("[^A-Za-z0-9\s]", "_", x) for x in X.columns]
    clf = lgb.LGBMClassifier().fit(X, y)
    feat_imp = pd.Series(dict(zip(X.columns, clf.feature_importances_))
                            ).sort_values().rename(yearInterval)
    importances.append(feat_imp)
    
dfCorrelations = pd.concat(correlations, axis=1).dropna()

dfImportances = pd.concat(importances, axis=1)

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.title("Pearson Correlations of the top-5 ICD chapters most predictive  of Death over time")
dfCorrelations.T.loc[:, dfCorrelations.abs().sum(axis=1).sort_values().tail(5).index
                     ].plot.line(ax=plt.gca())
plt.gca().legend(bbox_to_anchor=(0.9, -.08))
plt.subplot(1,2,2)
plt.title("Feature Importances of the top-5 ICD chapters most predictive  of Death over time")
dfImportances.T.loc[:,dfImportances.sum(axis=1).sort_values().tail(5).index
                    ].plot.line(ax=plt.gca())
plt.gca().legend(bbox_to_anchor=(0.8, -.08))