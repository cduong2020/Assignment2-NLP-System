from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load your existing knowledge base file
with open('data/knowledge_base.txt', 'r') as f:
    documents = f.readlines()
def baseline_retrieval(query, k=3):

    # engine that defines how to turn words into numbers (ignoring common 'stop_words' like 'the', 'is', etc.)
    vectorizer = TfidfVectorizer(stop_words='english')

    # sparse matrix representing the importance of every word across all LTI documents
    tfidf_matrix = vectorizer.fit_transform(documents)

    # vector representing the importance of keywords in the user's specific question
    query_vec = vectorizer.transform([query])


    # array of decimal numbers between 0 and 1 representing how much the query overlaps with each documents
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # position of the line with the highest similarity score
    best_index = np.argmax(scores)
    # Return only the single best string
    return documents[best_index]

results = []
with open('questions.txt', 'r') as file:
    for line in file:
        print("Question: " + line)
        print("Answer: " + baseline_retrieval(line))