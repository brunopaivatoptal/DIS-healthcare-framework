# -*- coding: utf-8 -*-
"""
Caracterização do MIMIC-IV
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

"""
emar = pd.read_csv(root/"hosp"/"emar.csv.gz")
"""
emar.columns
Index(['subject_id', 'hadm_id', 'emar_id', 'emar_seq', 'poe_id', 'pharmacy_id',
       'enter_provider_id', 'charttime', 'medication', 'event_txt',
       'scheduletime', 'storetime'],
      dtype='object')
"""
emar.loc[:,["event_txt", "medication"]]

omr = pd.read_csv(root/"hosp"/"omr.csv.gz")
"""
omr.columns
Out[63]: Index(['subject_id', 'chartdate', 'seq_num', 'result_name', 'result_value'], dtype='object')
"""

omr.loc[:,["result_name","result_value"]]


poe = pd.read_csv(root/"hosp"/"poe.csv.gz")
"""
poe.columns
Index(['poe_id', 'poe_seq', 'subject_id', 'hadm_id', 'ordertime', 'order_type',
       'order_subtype', 'transaction_type', 'discontinue_of_poe_id',
       'discontinued_by_poe_id', 'order_provider_id', 'order_status'],
      dtype='object')
"""

poe.loc[:,["poe_seq","poe_id"]]



drg = pd.read_csv(root/"hosp"/"drgcodes.csv.gz")
"""
drg.columns
Index(['subject_id', 'hadm_id', 'drg_type', 'drg_code', 'description',
       'drg_severity', 'drg_mortality'],
      dtype='object')
"""

drg.loc[:,["drg_code","description"]]
