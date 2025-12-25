from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer 


BASE_DIR = Path(__file__).resolve().parents[2]
CHROMA_DIR = BASE_DIR / "chroma_store"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model = None 
_collection = None 

def _get_model(): 
    global _model 
    if _model is None: 
        _model = SentenceTransformer(MODEL_NAME) 
    
    return _model 

def _get_collection(): 
    global _collection 
    if _collection is None: 
        client = chromadb.PersistentClient(path=str(CHROMA_DIR)) 
        _collection = client.get_or_create_collection(name="destinations")    #get access to destinations vector space stored in laption disk. its empty if u never run indexer first. this file would be idle . create for safety (no crash)
    
    return _collection 


def recommend(query: str , k: int = 1): 
    model = _get_model()
    col = _get_collection()
    
    query_emb = model.encode([query], normalize_embeddings=True).tolist()   
    res = col.query(                                                        #finding similar or close vectors to query through cosine sim
                    query_embeddings = query_emb,                            #list of vectors . in this case its just one 
                    n_results = k, 
                    include=["metadatas", "documents", "distances"]         #distances r stored auto
                    ) 
    
    out = [] 
    for md, doc, dist in zip(res["metadatas"][0], res["documents"][0], res["distances"][0]):
        out.append({
            "score": 1.0 - float(dist),  # convert distance -> similarity-ish
            "metadata": md,
            "document": doc,
        })
    return out

