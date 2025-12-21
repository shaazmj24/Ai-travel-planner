import pandas as pd 
from pathlib import Path 

# __file__ → backend/app/data_loader.py
# parents[0] → backend/app
# parents[1] → backend
# parents[2] → project root //parents[2] points to the project root directory then go into data folder...
BASE = Path(__file__).resolve().parents[2]
DATA_PATH = BASE / "data" / "destination.csv"

#Altenrative: DATA_PATH = "/Users/shaazmeghani/Ai-travel-planner/data/destination.csv"

def load_destination(): 
    return pd.read_csv(DATA_PATH) 



