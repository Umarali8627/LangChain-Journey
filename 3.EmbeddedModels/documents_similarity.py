from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np



embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# now create a documents 
documents =[
    "Islamabad is the capital of Pakistan",
    "Babar Azam is the captain of Pakistan Cricket Team and known for his classy batting style",
    "Shaheen shah is a greatbowler and belons from kpk province of Pakistan",
    "Fakhar Zaman is an aggressive batsman and plays for Pakistan Cricket Team",
    "Jasprit Bumrah is a fast bowler and plays for Indian Cricket Team and known for his unique bowling action",
    "Naseem Shah is the Fastest bowler and plays for Pakistan Cricket Team",
    "Umar Ali is the student of computer science and lives in akora Khatak pationated about AI/ML Engineer",
]

query ="tell me something about Umar Ali"
# get embeddings for documents
doc_embeddings =embeddings.embed_documents(documents)
# now get embedding for query
query_embedding =embeddings.embed_query(query)

# now compute cosine similarity between query and documents
similarities =cosine_similarity([query_embedding],doc_embeddings)[0]

index,score =sorted(list(enumerate(similarities)), key=lambda x: x[1], reverse=True)[0]

print("Query:", query)
print(f"Results : {documents[index]}\nwith Similarity score {score}")