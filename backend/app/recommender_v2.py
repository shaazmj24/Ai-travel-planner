import numpy as np 
import pandas as pd 
from scipy.spatial.distance import cdist  
from typing import Optional

VIBE = ["beach", "city", "nature", "party"]  
BUDGET = ["low", "mid", "high"] 
SEASON = ["winter", "spring", "summer", "fall"] 

def one_hot(value: str, categories: list[str]) -> np.ndarray:      #turning catrogical to numeric vector 
    v = np.zeros(len(categories), dtype=float) 
    if value is None: 
        return v 
    if value in categories: 
        v[categories.index(value)] = 1.0 
    return v 

def row_to_vector(row: pd.Series) ->np.ndarray:               #turning existing destination from dataset to vector 
    return np.concatenate([ 
                           one_hot(row.get("vibe"), VIBE), 
                           one_hot(row.get("budget"), BUDGET),  
                           one_hot(row.get("best_season"), SEASON),  
                           ]) 

def prefs_to_vector(vibe: Optional[str], budget: Optional[str], season: Optional[str]) -> np.ndarray: #user query turning into R^11 vector 
    return np.concatenate([    
                           one_hot(vibe, VIBE), 
                           one_hot(budget, BUDGET), 
                           one_hot(season, SEASON),
                           ])  
    

def recommend(df: pd.DataFrame, vibe: Optional[str] = None, budget: Optional[str] = None, season: Optional[str] = None, k: int = 5) -> pd.DataFrame:       #intital none if para missing. without none, treat it like requiremnt which will crash if para missin
    df = df.copy() 
    X = np.vstack([row_to_vector(r) for _, r in df.iterrows()])   # (n x d(dimesion)) , iterrow: row by row logic. _ is index, r is row. skip index para 
    q = prefs_to_vector(vibe, budget, season).reshape(1, -1)      # (1 x d) , turning general vector or 1d into 2d array / row vector

    # cosine distance → smaller is better
    dist = cdist(q, X, metric="cosine").flatten()                 #q ->(1, Features) X -> (n, Features) n rows in dataset . cdist requires 2d array and returns 2d array where each row represents cosine simialirty between q and Xi . flatten turns 2d into 1d. pandas wanst 1d arraus
    df["score"] = 1.0 - dist   # bigger consine better. assign one score per row by index (vectroized assignment). dist is 1d array 

    df = df.sort_values(["score", "name"], ascending=[False, True]) 
    
    return df.head(k)
    



#destination (text/categorical)
#       ↓
#numeric vector
#       ↓
#user preference vector
#       ↓
#cosine similarity
#       ↓
#rank top K



