import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        # use the same translator as in ingest.py
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # loads the index (maps, directions)
        self.index = faiss.read_index("data/docs.index")
        # loads the human-readable text
        with open("data/chunks.txt", "r") as f:
            self.chunks = f.readlines()

    def search(self, query, k=3):
        # turns the query into the 384D vector
        query_vec = self.model.encode([query])
        # identified the nearest k points to the query_vec point
        distances, indices = self.index.search(np.array(query_vec), k)
        # grabs and returns the human-readable text at the location given by indices
        return [self.chunks[i] for i in indices[0]]