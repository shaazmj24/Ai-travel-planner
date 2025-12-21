import numpy as np 
import pandas as pd 
from scipy.spatial.distance import cdist 

VIBE = ["beach", "city", "nature", "party"]  
BUDGET = ["low", "mid", "high"] 
SEASON = ["winter", "spring", "summer", "fall"] 

def one_hot (value: str, categories: list[str]) -> np.ndarray:      #turning catrogical to numeric vector 
    v = np.zeros(len(categories), dtype=float) 
    if value is None: 
        return v 
    if value in categories: 
        v[categories.index(value)] = 1.0 
    return v 

def row_to_vector(row: pd.Series) ->np.ndarray:               #turning existing destination from dataset to vector 
    return np.concatenate([ 
                           one_hot(row.get("vibe", VIBE)), 
                           one_hot(row.get("budget"), BUDGET),  
                           one_hot(row.get("best_season"), SEASON),  
                           ]) 

def prefs_to_vector(vibe: str | None, budget: str | None, season: str | None) -> np.ndarray: #user query turning into R^11 vector 
    return np.concatenate([    
                           one_hot(vibe, VIBE), 
                           one_hot(budget, BUDGET), 
                           one_hot(season, SEASON),
                           ])  
    

def recommend(df: pd.DataFrame, vibe: str | None, budget: str | None, season: str | None, k: int = 5) -> pd.DataFrame: 
    df = df.copy() 
    X = np.vstack([row_to_vector(r) for _, r in df.iterrows()])   # (n x d(dimesion)) , iterrow: row by row logic. _ is index, r is row. skip index para 
    q = prefs_to_vector(vibe, budget, season).reshape(1, -1)      # (1 x d) , turning general vector or 1d into 2d array / row vector

    # cosine distance → smaller is better
    dist = cdist(q, X, metric="cosine").flatten()
    df["score"] = 1.0 - dist   # cosine similarity → bigger is better

    df = df.sort_values(["score", "name"], ascending=[False, True]) 
    
    return df.head(k)
    

