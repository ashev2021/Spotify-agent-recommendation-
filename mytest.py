import faiss
import numpy as np
import json
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app1 = FastAPI()
model = SentenceTransformer('facebook/musicbert-base')
index = faiss.IndexFlatL2(387)
song_meta =[]

songs = [
    {"title": "Bad Guy", "artist": "Billie Eilish", "description": "dark, pop, energetic"},
    {"title": "Blinding Lights", "artist": "The Weeknd", "description": "retro, 80s, synth-pop"},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "description": "classic, rock, operatic"},
    {"title": "Shape of You", "artist": "Ed Sheeran", "description": "romantic, acoustic, pop"},
    {"title": "Lose Yourself", "artist": "Eminem", "description": "rap, motivational, intense"},
]

for song in songs:
    embedding = model.encode(song["description"])
    index.add(np.array([embedding]))
    song_meta.append({"title": song["title"], "artist": song["artist"]})

class SongQuery(BaseModel):
    description: str

@app1.post("/recommend")
def recommend_song(query: SongQuery):
    query_vec = model.encode(query.description)
    D, I = index.search(np.array([query_vec]), k=3)
    results = [song_meta[i] for i in I[0]]
    return {"recommendations": results}
