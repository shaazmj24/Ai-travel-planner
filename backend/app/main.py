from fastapi import FastAPI 
from data_loader import load_destination 
from recommender_v1 import recommend as recommend_v1


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
def recommend_endpoint(vibe: str, budget: str, season: str, k: int = 5): 
    df = load_destination() 
    out = recommend_v1(df, vibe=vibe, budget=budget, season=season, k=k)  
    return out.to_dict(orient="records")
    
