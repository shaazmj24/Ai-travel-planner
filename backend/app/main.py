from fastapi import FastAPI 
from data_loader import load_destination 
from recommender_v1 import recommend as recommend_v1 
from recommender_v2 import recommend as recommend_v2 
from typing import Optional 
from recommender_v4 import recommend as recommend_v4


app = FastAPI() #starts n creates web application. app being api server 

@app.get("/") 
def f(): 
    return {"status: ok"}

@app.get("/destinations")    #base
def destination(): 
    df = load_destination() 
    return df.to_dict(orient="records") #orient: shape of output 


@app.get("/recommend")      #base
def recommend(vibe: str, budget: str, season: str): 
    df = load_destination()   
    
    filtered = df[ 
                  (df["vibe"] == vibe) &  
                  (df["budget"] == budget) & 
                  (df["best_season"] == season)
                  ] 
    
    return filtered.to_dict(orient="records")

@app.get("/recommend_v1")   #top k suggested destination 
def recommend_endpoint(vibe: str, budget: str, season: str, k: int): 
    df = load_destination() 
    out = recommend_v1(df, vibe=vibe, budget=budget, season=season, k=k)  
    return out.to_dict(orient="records")
    
@app.get("/recommend_v2")  #top k suggestd destination using consine similairy 
def recommend_v2_endpoint(vibe: Optional[str] = None, budget: Optional[str] = None, season: Optional[str] = None, k: int = 0): 
    df = load_destination() 
    out = recommend_v2(df, vibe=vibe, budget=budget, season=season, k=k) 
    return out.to_dict(orient="records") 

@app.get("/recommend_v4")   #top k suggested destination using embedding model along with cosine sim
def recommend_v4_endpoint(query: str, k: int = 1):
    return recommend_v4(query=query, k=k)


