# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:21:02 2023

@author: angel
"""

from multiprocessing import Pool, cpu_count 
from tqdm import tqdm
import pandas as pd

try:
    num_cpus = min(cpu_count(), 63)
except:
    num_cpus=2
    
icdChapter = pd.read_excel("icd-block-classification.xlsx")
icd_memory = {}

def estimateAllCorrelationsParallell(df, deaths):
    pool = Pool(num_cpus)
    corrList = pool.starmap(estimateCorrelation, [(df[c], c, deaths) for c in tqdm(df.columns)])
    return corrList
    
def estimateCorrelation(dummyCol : pd.Series, c : str, deaths : pd.Series):
    """
    Inputs : two identically indexed series and a string
    """
    corr = pd.concat([dummyCol, deaths], axis=1).corr().loc[c,"death"]
    return {c: corr}


def classifyICD_Code(icd, icdChapter=icdChapter):
    try:
        if icd in icd_memory:
            return icd_memory[icd]
        letter = icd[0]
        numbers = int(icd[1:3])
        
        N = ord(letter) * 100 + numbers
        ck = icdChapter.copy()
        ck["range_start"] = ck.RANGE_START.apply(lambda x: ord(x[0]) * 100 + int(x[1:3]))
        ck["range_end"] = ck.RANGE_END.apply(lambda x: ord(x[0]) * 100 + int(x[1:3]))
        overlap = ck[(ck.range_start <= N) & (ck.range_end >= N)].CHAPTER.values[0]
        icd_memory.update({icd:overlap})
        return overlap
    except:
        return "N/A"