from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

documents = [
    "Kendrick Lamar is a Pulitzer Prize-winning rapper known for his socially conscious lyrics and storytelling.",
    "Drake is a Canadian rapper and singer who blends hip-hop and R&B in his emotionally driven music.",
    "J. Cole is a rapper and producer recognized for his introspective lyrics and self-produced albums.",
    "Travis Scott is known for his psychedelic trap sound and high-energy performances.",
    "Nicki Minaj is a Trinidadian-American rapper famed for her sharp wordplay and vibrant alter egos."
]

query = "tell me about j cole"

embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions='100')

doc_embedding = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(query)

similarity = cosine_similarity([query_embedding], doc_embedding)[0]

index, score = (sorted(list(enumerate(similarity)), key= lambda x:x[1])[-1])

print(query)
print(documents[index])
print("Similarity Score is:", score)