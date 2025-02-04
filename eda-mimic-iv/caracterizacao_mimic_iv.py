# -*- coding: utf-8 -*-
"""
Caracterização do MIMIC-IV na tarefa de predição de óbito
"""

from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import os


root = Path("../../DATASETS/MIMIC-IV/mimic-iv-2.2")

patients = pd.read_csv(root/"hosp"/"patients.csv.gz")
"""
patients.columns
Index(['subject_id', 'gender', 'anchor_age', 'anchor_year',
       'anchor_year_group', 'dod'],
      dtype='object')
"""


icdSubject = pd.read_csv(root/"hosp"/"diagnoses_icd.csv.gz")
icdDiagn = pd.read_csv(root/"hosp"/"d_icd_diagnoses.csv.gz")
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

## Lethality over time
data = {}
for y in tqdm(patientsWithIcds.anchor_year_group.unique()):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == y].copy()
    deaths = (~view.dod.isna()).value_counts(normalize=True).to_dict()
    data.update({y:{'proportion of deaths':deaths[True], 'proportion of non-deaths':deaths[False]}})
    
plt.figure(figsize=(12,6))
plt.title("Lethality over anchor_year_group")
dfLethality = pd.DataFrame(data).loc[:,['2008 - 2010', '2011 - 2013', '2014 - 2016','2017 - 2019']].copy()
dfLethality.T.plot.barh(stacked=True, color=["crimson","g"], ax=plt.gca())

## Most common diseases over time
mostCommon = {}
allDiseases = {}
for y in tqdm(patientsWithIcds.anchor_year_group.unique()):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == y].copy()
    print(view.long_title.value_counts(normalize=True).rename(y).head(10).reset_index())
    commonDiseases = view.long_title.value_counts(normalize=True)
    mostCommon.update({y:commonDiseases.head(5).to_dict()})
    allDiseases.update({y:commonDiseases.to_dict()})
    
plt.figure(figsize=(12,10))
plt.title("Most common diseases over time")
dfLDiseases = pd.DataFrame(mostCommon).loc[:,['2008 - 2010', '2011 - 2013', '2014 - 2016','2017 - 2019']].T.copy()
dfLDiseases.plot.barh(stacked=False, ax=plt.gca())



## Pearson correlations between diseases and death (TOP-5)
import Utils
mostCommonCorr = {}
allCorr = {}
for y in tqdm(patientsWithIcds.anchor_year_group.unique()):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == y].copy()
    dummies = pd.get_dummies(view.long_title)
    deaths = (~view.dod.isna()).astype(float).rename("death")
    
    corrDict = {}
    for c in tqdm(dummies.columns):
        corr = pd.concat([dummies[c], deaths], axis=1).corr().loc[c, "death"]
        corrDict.update({c:corr})
    
    corrDictTop10 = pd.Series(corrDict).sort_values(ascending=False).head(10).to_dict()
    mostCommonCorr.update({y:corrDictTop10})
    allCorr.update({y:corrDict})
    
plt.figure(figsize=(15,10))
plt.title("Diseases with the highest correlation to deaths over time")
dfCorrDeath = pd.DataFrame(mostCommonCorr).loc[:,['2008 - 2010', '2011 - 2013', '2014 - 2016','2017 - 2019']].T.copy()
allCorrDf = pd.DataFrame(allCorr)
sns.heatmap(allCorrDf.T.loc[:,dfCorrDeath.columns].T, cmap="coolwarm", annot=True)

allCorrDf.to_parquet("data/all_disease_correlations_to_death.gzip")


top5ColsL = [allCorrDf[s].sort_values(ascending=False).head(5).index for s in allCorrDf.columns]
uniqueCOls = sorted(list(set([x for y in top5ColsL for x in y])))

plt.figure(figsize=(15,10))
plt.title("Diseases with the highest correlation to deaths over time")
allCorrDf.loc[uniqueCOls,['2008 - 2010', '2011 - 2013', '2014 - 2016','2017 - 2019']].T.plot.barh(ax=plt.gca())


## Parallell coordinates:
plt.figure(figsize=(12,6))
plt.title("Correlation with death of the top-500 most common diseases")
mostCommonDiseasesTopK = patientsWithIcds.long_title.value_counts().head(500)
allCorrDf.loc[mostCommonDiseasesTopK.index, :].T.plot.line(alpha=0.2, ax=plt.gca(), legend=False, color="crimson")


## feature importances over time
import lightgbm as lgb
import Utils

allFeatureImport = {}
for years in tqdm(patientsWithIcds.anchor_year_group.unique()):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == years].copy()
    dummies = pd.get_dummies(view.long_title.str.replace("[^A-Za-z0-9\s]", "_"))
    deaths = (~view.dod.isna()).astype(float).rename("death")

    X, y = dummies, deaths    
    clf = lgb.LGBMClassifier(n_jobs=-1).fit(X, y)
    
    featureImp_dict = pd.Series(clf.feature_importances_, index=X.columns).sort_values().to_dict()
    allFeatureImport.update({years:featureImp_dict})
    
allFeatureImportDf = pd.DataFrame(allFeatureImport)
allFeatureImportDf.to_parquet("data/all_disease_feature_import_over_time.gzip")

plt.figure(figsize=(12,6))
plt.title("Feature importances of all diseases when predicting death accross time")
allFeatureImportDf.T.plot.line(alpha=0.2, ax=plt.gca(), legend=False, color="crimson")


mostCommonDiseasesTopK = patientsWithIcds.long_title.str.replace("[^A-Za-z0-9\s]", "_").value_counts().head(1_000)
mostCommonFeatImport = allFeatureImportDf.loc[mostCommonDiseasesTopK.index, :].T.copy()
delta = mostCommonFeatImport.max() - mostCommonFeatImport.min()

deltaPrnt = delta.copy()
deltaPrnt.index = [x[:40] for x in deltaPrnt.index]
deltaPrnt.sort_values(ascending=False).head(20)

plt.figure(figsize=(12,6))
plt.title("Feature importances the top-1000 most common diseases when predicting death accross time")
mostCommonFeatImport.T.plot.line(alpha=0.2, ax=plt.gca(), legend=False, color="crimson")


## Using colormap
import matplotlib.pylab as pl
colors = pl.cm.cool(np.linspace(0,1,delta.astype(int).max()))
deltaDict = delta.astype(int).to_dict()

orderedYears = ['2008 - 2010', '2011 - 2013', '2014 - 2016','2017 - 2019']
plt.figure(figsize=(12,6))
plt.title("Feature importances the top-1000 most common diseases when predicting death accross time")
for c in tqdm(delta.sort_values().index):
    mostCommonFeatImport[c][orderedYears].plot.line(ax=plt.gca(), color=colors[deltaDict[c] - 1], alpha=0.4)
plt.ylabel("Feature importances")
plt.xlabel("Anchor year group")
