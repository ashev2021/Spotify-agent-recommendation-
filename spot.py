# === SETUP ===
# You need: spotipy, sentence-transformers, faiss-cpu, langchain, openai, fastapi, uvicorn

import faiss
import numpy as np
import json
from fastapi import FastAPI, Query
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.IndexFlatL2(384)
song_metadata = []  # List to store song info alongside embeddings

# === DUMMY DATA ===
# Normally you'd crawl this using Spotify API
songs = [
    {"title": "Bad Guy", "artist": "Billie Eilish", "description": "dark, pop, energetic"},
    {"title": "Blinding Lights", "artist": "The Weeknd", "description": "retro, 80s, synth-pop"},
    {"title": "Bohemian Rhapsody", "artist": "Queen", "description": "classic, rock, operatic"},
    {"title": "Shape of You", "artist": "Ed Sheeran", "description": "romantic, acoustic, pop"},
    {"title": "Lose Yourself", "artist": "Eminem", "description": "rap, motivational, intense"},
]

for song in songs:
    embedding = model.encode(song['description'])
    index.add(np.array([embedding]))
    song_metadata.append({"title": song["title"], "artist": song["artist"]})

# === QUERY INTERFACE ===
class SongQuery(BaseModel):
    description: str

@app.post("/recommend")
def recommend_song(query: SongQuery):
    query_vec = model.encode(query.description)
    D, I = index.search(np.array([query_vec]), k=3)
    results = [song_metadata[i] for i in I[0]]
    return {"recommendations": results}

# === RUN ===
# Run with: uvicorn spotify_rec_agent:app --reload
# Then POST to http://127.0.0.1:8000/recommend with JSON: {"description": "energetic pop"}

if __name__ == "__main__":
    import uvicorn
    # Change host to "0.0.0.0" to allow external access
    uvicorn.run(app, host="0.0.0.0", port=8000)
