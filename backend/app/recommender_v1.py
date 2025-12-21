import pandas as pd 

def score_destination(row: pd.Series, vibe: str, budget: str, season: str) -> int:        #pd.series is just 1d srtucture in this case  {vibe : ...., season : ...}
    score = 0 
    
    if str(row.get("vibe", "")).lower() == vibe.lower(): 
        score += 1 
    if str(row.get("budget", "")).lower() == budget.lower(): 
        score += 1  
    if str(row.get("best_season", "")).lower() == season.lower():
        score += 1
    
    return score 

def recommend(df: pd.DataFrame, vibe: str, budget: str, season: str, k: int = 5) -> pd.DataFrame: 
    df = df.copy() 
    df["score"] = df.apply(lambda r: score_destination(r, vibe, budget, season), axis = 1)   #generally dt[new_column] = value but row by row u use apply. axis = 1 for row axis = 0 for column
    df = df.sort_values(["score", "name"], ascending=[False, True])                          #score descending if scores same then sort name by aplhebet order in ascending order 
    return df.head(k) 


    
    
    
    
    
    
    
    
    
    
    