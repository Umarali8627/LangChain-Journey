from langchain_huggingface import HuggingFaceEmbeddings
# embeddings for a just a single query
# using sentence-transformers model opensource model 

embeddings =HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text="Islanabad is the capital of Pakistan"

vector=embeddings.embed_query(text)
print(str(vector))