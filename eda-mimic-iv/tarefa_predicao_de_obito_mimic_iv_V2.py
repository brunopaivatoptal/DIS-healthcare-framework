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
expl = icds.groupby(by=["icd_code"]).long_title.first().to_dict()

patientsWithIcds = pd.merge(patients, icds, on=["subject_id"])
icds_to_exclude = ["Z66",  ## Do not resuscitate
                   "Z515", ## hospitalization for palliative care
                   "R570", ## Cardiogenic shock
                   ## different types of shock
                   'R6520', 'R579', 'R570', 'R578', 'R6521', 'R571', 'T8119XA',
                   'T794XXA', 'T8111XA', 'T8112XA', 'T882XXA', 'T782XXA', 'A483',
                   'O751', 'T8110XA', 'O083', 'Y843', 'T782XXS',
                   ## death ICDs
                   'Z8241', 'S065X7A', 'S06357A', 'Z634', 'G9382', 'S061X7A',
                   'O3123X2', 'S066X7A', 'S062X7A', 'O364XX0', 'O3122X1', 'S065X8A',
                   'S066X8A', 'S069X7A', 'O3123X1', 'S06897A', 'S064X7A', 'S06377A',
                   'O3121X1', 'O3123X0', 'S061X8A', 'S06347A'
                   ]
patientsWithIcds = patientsWithIcds[~patientsWithIcds.icd_code.isin(icds_to_exclude)].copy()


testYears = ['2017 - 2019', '2014 - 2016', '2011 - 2013']
trainAnchorYear = "2008-2010"
fullData_l = []
trainView = patientsWithIcds[patientsWithIcds.anchor_year_group == '2008 - 2010'].copy()
trainView["obito"] = (~trainView.dod.isna()).astype(int)
gb = trainView.groupby(by=["subject_id"])
X_raw = gb.icd_code.apply(list).apply(lambda x: " ".join(x))
y_raw = gb.obito.max()
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw)

vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
idx = pd.Series({vectorizer.vocabulary_[x]: x for x in vectorizer.vocabulary_}).sort_values()
X_train_vect_n = pd.DataFrame(X_train_vect.toarray(), columns=idx.values).astype(np.float32)
clf = lgb.LGBMClassifier(class_weight="balanced").fit(X_train_vect_n, y_train)

metrics = []

for yearInterval in tqdm(testYears):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    view["obito"] = (~view.dod.isna()).astype(int)
    gb = view.groupby(by=["subject_id"])
    X_raw = gb.icd_code.apply(list).apply(lambda x: " ".join(x))
    y_raw = gb.obito.max()
        
    X_test_vect = vectorizer.transform(X_raw)
    X_test_vect_n = pd.DataFrame(X_test_vect.toarray(), columns=idx.values).astype(np.float32)
    y_pred = clf.predict(X_test_vect_n)
    
    metrics.append({
        "macroF1":f1_score(y_raw, y_pred, average="macro"),
        "microF1":f1_score(y_raw, y_pred, average="micro"),
        "recall":recall_score(y_raw, y_pred),
        "year":yearInterval
        })

dfMetrics = pd.DataFrame([pd.Series(m) for m in metrics]).sort_values(by=["year"])
print(dfMetrics)

plt.figure(figsize=(10,5))
plt.title("Death classification metrics using diseases over time periods")
dfMetrics.set_index("year").plot.bar(ax=plt.gca())
plt.xticks(rotation=0)



plt.figure(figsize=(6,6))
lgb.plot_importance(clf, max_num_features=30, ax=plt.gca())

"""
expl["F0390"]
Out[14]: 'Unspecified dementia without behavioral disturbance'

expl["R570"]

"""