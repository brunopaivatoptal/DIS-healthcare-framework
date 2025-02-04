# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 19:22:05 2023

@author: angel
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


root = Path("../../../DATASETS/MIMIC-IV/mimic-iv-2.2")

patients = pd.read_csv(root/"hosp"/"patients.csv.gz")

patients.columns



icdSubject = pd.read_csv(root/"hosp"/"diagnoses_icd.csv.gz")
icdDiagn = pd.read_csv(root/"hosp"/"d_icd_diagnoses.csv.gz")
icdDiagn = icdDiagn[icdDiagn.icd_version == 10].copy()

icds = pd.merge(icdSubject, icdDiagn, on=["icd_code", "icd_version"]).sort_values(by=["subject_id","seq_num"])


patientsWithIcds = pd.merge(patients, icds, on=["subject_id"])

fullData_l = []

for yearInterval in tqdm(patientsWithIcds.anchor_year_group.unique()):
    view = patientsWithIcds[patientsWithIcds.anchor_year_group == yearInterval].copy()
    icdList = view.sort_values(by=["subject_id", "seq_num"]).groupby(by=["subject_id"]).long_title.apply(list)
    w2v = Word2Vec(sentences=icdList.values.tolist(), vector_size=100)
    
    vecs = np.vstack([w2v.wv[x] for x in tqdm(w2v.wv.index_to_key)])
    dst = cosine_distances(vecs, vecs)
    df_i = pd.DataFrame(dst, columns=w2v.wv.index_to_key, index=w2v.wv.index_to_key).reset_index().rename(columns={"index":"diagnosis"})
    df_i["anchor_year_group"] = yearInterval
    fullData_l.append(df_i)    

fullData = pd.concat(fullData_l).reset_index(drop=True)
naPrevalence = fullData.isna().mean(axis=0).sort_values()
onAllYears = naPrevalence[naPrevalence == 0]
fullData = fullData.loc[:,onAllYears.index].copy()


X_raw = fullData.drop(columns=["diagnosis","anchor_year_group"])
tsne = TSNE(n_components=2, n_jobs=-1)
X_embedded = tsne.fit_transform(X_raw)

fullData["X1"] = X_embedded[:,0]
fullData["X2"] = X_embedded[:,1]

icdDict = icdDiagn.set_index("long_title").icd_code.to_dict()
fullData["Chapter"] = [icdDict[x][0] for x in tqdm(fullData.diagnosis)]

import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Create figure
fig = go.Figure()

anchorYears = fullData.anchor_year_group.unique()
# Add traces, one for each slider step
for step in tqdm(anchorYears):
    view = fullData[fullData.anchor_year_group == step].copy()
    view["text"] = view.diagnosis + "- Chapter :" + view.Chapter
    fig.add_trace(
        go.Scatter(
            visible=False,
            mode="markers",
            hoverinfo=["text"],
            text=view.text.values,
            marker=dict(color=view.Chapter.apply(lambda x: ord(x))),
            x=view["X1"],
            y=view["X2"]))

# Make 10th trace visible
fig.data[0].visible = True

# Create and add slider
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": anchorYears[i]}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Frequency: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

with open("plot-yearly-evolution-of-disease-similarity.html","w") as f:
    html = fig.to_html()
    f.write(html)
    
os.system("plot-yearly-evolution-of-disease-similarity.html")


## How much diseases changed over the years
dropCOls = ['X1', 'X2', 'Chapter', 'anchor_year_group']
explored = set()
changeRates = []
for yi in tqdm(fullData.anchor_year_group.unique()):
    view1 = fullData[fullData.anchor_year_group == yi].copy().set_index("diagnosis").drop(columns=dropCOls)
    for yj in fullData.anchor_year_group.unique():
        if yi != yj and "->".join(sorted([yi,yj])) not in explored:
            view2 = fullData[fullData.anchor_year_group == yj].copy().set_index("diagnosis").drop(columns=dropCOls)
            for disease in tqdm(view1.index):
                if disease in view2.index:
                    dst = cosine_distances([view1.loc[disease]], [view2.loc[disease]])
                    changeRates.append(pd.Series({
                        "from-years":yi,
                        "to-years":yj,
                        "diagnosis":disease,
                        "pattern-change":dst.reshape(-1)[0]
                        }))
                explored = explored.union(set(["->".join(sorted([yi,yj]))]))
                    
dfChange = pd.concat(changeRates, axis=1).T
dfChange = dfChange.sort_values(by=["pattern-change"], ascending=False)

plt.figure(figsize=(10,10))
plt.title("Diseases with biggest co-occurrence change")
dfChange.head(20).sort_values(by=["pattern-change"]).set_index("diagnosis").plot.barh(ax=plt.gca())

for disease, fromYears, toYears in tqdm(dfChange.loc[:,['diagnosis', 'from-years', 'to-years']].head(20).values):
    fromView = fullData[(fullData.diagnosis == disease) & (fullData["anchor_year_group"] == fromYears)].copy().drop(columns=["Chapter","diagnosis","X1","X2"])
    toView = fullData[(fullData.diagnosis == disease) & (fullData['anchor_year_group'] == toYears)].copy().drop(columns=["Chapter","diagnosis","X1","X2"])
    
    changeDirection = sorted([fromYears, toYears])
    dfCompare = pd.concat([fromView.T.iloc[:,0].rename(fromYears),
                    toView.T.iloc[:,0].rename(toYears)], axis=1
                ).sort_values(by=[changeDirection[0], changeDirection[1]]
                              ).drop(index=["anchor_year_group"]).loc[:,changeDirection]
    
    plt.figure(figsize=(8,12))
    plt.subplot(2,1,1)
    plt.title(disease)
    dfCompare.tail(10).plot.barh(ax=plt.gca())
    plt.subplot(2,1,2)
    dfCompare.head(11).iloc[1:].plot.barh(ax=plt.gca())