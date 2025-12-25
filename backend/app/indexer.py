from pathlib import Path 
import chromadb 
import pandas as pd 
from sentence_transformers import SentenceTransformer 

#intilizing chromaBD vectordatabase 

BASE_DIR = Path(__file__).resolve().parents[2] 
DATA_PATH = BASE_DIR / "data" / "destination.csv" 
CHROMA_DIR = BASE_DIR / "chroma_store" 
 
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_text(row: pd.Series) -> str: 
    parts = [  
             str(row.get("name", "")), 
             str(row.get("country", "")), 
             str(row.get("description", "")), 
             f"vibe:{row.get('vibe','')}", 
             f"budget:{row.get('budget','')}", 
             f"season:{row.get('best_season','')}",
    ] 
    return " | ".join([p for p in parts if p and p != "nan"]).strip()    #embedding model performs better with delimter (structured) 

def main(): 
    df = pd.read_csv(DATA_PATH) 
    model = SentenceTransformer(MODEL_NAME)  
    docs = [build_text(r) for _, r in df.iterrows()]   #turn row into organised docs seperated by delimter | 
    
    ids = [str(i) for i in range(len(df))]             #retuen no. of rows 
    
    metadatas = [] 
    for _, r in df.iterrows(): 
        metadatas.append({     
                        "name": str(r.get("name", "")), 
                        "country": str(r.get("country", "")), 
                        "vibe": str(r.get("vibe", "")), 
                        "budget": str(r.get("budget", "")), 
                        "best_season": str(r.get("best_season", "")), 
                        })  
    
    #applying embedding model to every row from dataset 
    embeddings = model.encode(docs, normalize_embeddings=True).tolist()   #it returns numpy array then u pass to toList to get [ [v1] , [v2], ... ] 
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))              #intialize connection. this database is now stored in laption disk 
    try:  #for safety
        client.delete_collection("destination") 
    except Exception: 
        pass   
    col = client.get_or_create_collection(name="destinations")            #name of vector space / database / table  
    col.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)   #adding info to vector database. embeddings contains numeric vectors they are stored in vector space 
     
if __name__ == "__main__": 
    main() 
    
    

    
    
